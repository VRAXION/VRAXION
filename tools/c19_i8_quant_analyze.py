#!/usr/bin/env python3
"""c19_i8_quant_analyze.py - side analysis on int8 quant of c19 per-neuron LUTs.

NO changes to existing tools / rust. Reads:
  target/c19_parity_sweep/seed_{04,09}/final.json
  target/c19_manual_grow/grid3_center/state.json
  target/c19_grower_smoke_parity/final.json

Outputs:
  - stdout: per-network tables and per-neuron deltas
  - optional --json <path>: dumps all measurements for downstream reporting
  - optional --report: prints a markdown block ready to paste into the report

Key design: we must replicate the RUST c19_grower predict() semantics:
    score = sum( alpha * lut_out )         [real-valued]
    hidden_sig = 1 if lut_out >= 0 else 0  [thresholded]
not the python manual-explorer one that uses alpha * sign(lut_out). The rust
ensemble_train/val/test numbers stored in final.json are the ground truth.

Data generation is a direct port of c19_grower.rs gen_data() + its FONT9 +
grid3_full_parity label_fn + LCG Rng. We parity-check our reimplementation
against the stored final.json numbers before running any quant study.

Usage:
    python tools/c19_i8_quant_analyze.py             # full analysis
    python tools/c19_i8_quant_analyze.py --verbose   # more detail
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Port of c19_grower.rs Rng (LCG, NOT xorshift)
# -----------------------------------------------------------------------------

MASK64 = (1 << 64) - 1
LCG_MUL = 6364136223846793005
LCG_ADD = 1442695040888963407


class LcgRng:
    def __init__(self, seed: int) -> None:
        # Rng::new(seed) = { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) }
        self.s = (seed * LCG_MUL + 1) & MASK64

    def next_u64(self) -> int:
        self.s = (self.s * LCG_MUL + LCG_ADD) & MASK64
        return self.s

    def next_f32(self) -> float:
        # fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
        return float(((self.next_u64() >> 33) % 65536)) / 65536.0

    def bool_p(self, p: float) -> bool:
        return self.next_f32() < p


FONT = [
    [1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1],
]


def label_grid3_center(px):
    return px[4]


def label_grid3_full_parity(px):
    return (px[0] ^ px[1] ^ px[2] ^ px[3] ^ px[4] ^ px[5] ^ px[6] ^ px[7] ^ px[8]) & 1


def gen_data(task: str, data_seed: int = 42, noise: float = 0.10, n_per: int = 200):
    """Port of c19_grower.rs gen_data(). Returns (train_X, train_y, val_X, val_y, test_X, test_y)."""
    if task == "grid3_center":
        label_fn = label_grid3_center
    elif task == "grid3_full_parity":
        label_fn = label_grid3_full_parity
    else:
        raise ValueError(f"unsupported task {task}")
    rng = LcgRng(data_seed)
    tr_x, tr_y = [], []
    va_x, va_y = [], []
    te_x, te_y = [], []
    for d in range(10):
        for i in range(n_per):
            px = list(FONT[d])
            for p_idx in range(len(px)):
                if rng.bool_p(noise):
                    px[p_idx] = 1 - px[p_idx]
            lab = label_fn(px)
            mod = i % 5
            if mod == 0:
                va_x.append(px); va_y.append(lab)
            elif mod == 1:
                te_x.append(px); te_y.append(lab)
            else:
                tr_x.append(px); tr_y.append(lab)
    return tr_x, tr_y, va_x, va_y, te_x, te_y


# -----------------------------------------------------------------------------
# Neuron eval (Rust semantics)
# -----------------------------------------------------------------------------

def neuron_dot(neuron: dict, sigs: list[int]) -> int:
    d = 0
    for w, p in zip(neuron["weights"], neuron["parents"]):
        if w != 0:
            d += int(w) * int(sigs[p])
    return d


def eval_lut(neuron: dict, sigs: list[int]) -> float:
    d = neuron_dot(neuron, sigs)
    idx = d - neuron["lut_min_dot"]
    if idx < 0:
        return 0.0
    if idx >= len(neuron["lut"]):
        return 0.0
    return float(neuron["lut"][idx])


def eval_lut_pred(neuron: dict, sigs: list[int]) -> int:
    return 1 if eval_lut(neuron, sigs) >= 0.0 else 0


def predict_rust(neurons: list[dict], inp: list[int]) -> int:
    """Mirror of Rust Net::predict() - score sums alpha*real_lut_out."""
    if not neurons:
        return 0
    sigs = list(inp)
    score = 0.0
    for n in neurons:
        v = eval_lut(n, sigs)
        score += float(n["alpha"]) * v
        sigs.append(1 if v >= 0.0 else 0)
    return 1 if score >= 0.0 else 0


def score_rust(neurons: list[dict], inp: list[int]) -> float:
    """Return the real-valued ensemble score (before thresholding at 0)."""
    sigs = list(inp)
    score = 0.0
    for n in neurons:
        v = eval_lut(n, sigs)
        score += float(n["alpha"]) * v
        sigs.append(1 if v >= 0.0 else 0)
    return score


def accuracy_rust(neurons: list[dict], X: list[list[int]], y: list[int]) -> float:
    if not neurons:
        return 50.0
    ok = 0
    for xi, yi in zip(X, y):
        if predict_rust(neurons, xi) == yi:
            ok += 1
    return 100.0 * ok / len(X)


# -----------------------------------------------------------------------------
# Quantizer
# -----------------------------------------------------------------------------

@dataclass
class QuantResult:
    scale: float
    zero_point: float
    q: list[int]
    lut_deq: list[float]
    max_abs_err: float
    l1_err: float
    identity_preserved: bool  # whether c19(0)=0 mapping is preserved
    zero_idx: int


def quantize_symmetric_absmax(lut: list[float], nbits: int = 8) -> QuantResult:
    """Symmetric int quant using absmax scale (scale = max(|lut|) / qmax).

    For nbits=8: qmax=127, qmin=-127.
    For nbits=4: qmax=7,   qmin=-7.
    Degenerate: all-zero LUT -> scale=1.0, q=[0...], lut_deq=[0...].
    """
    qmax = (1 << (nbits - 1)) - 1  # 127 for 8bit, 7 for 4bit
    qmin = -qmax
    absmax = max((abs(v) for v in lut), default=0.0)
    if absmax == 0.0:
        q = [0] * len(lut)
        deq = [0.0] * len(lut)
        zero_idx = len(lut) // 2  # not meaningful but consistent
        return QuantResult(
            scale=1.0, zero_point=0.0, q=q, lut_deq=deq,
            max_abs_err=0.0, l1_err=0.0,
            identity_preserved=True, zero_idx=zero_idx,
        )
    scale = absmax / qmax
    q = []
    for v in lut:
        qi = int(round(v / scale))
        if qi > qmax:
            qi = qmax
        elif qi < qmin:
            qi = qmin
        q.append(qi)
    deq = [qi * scale for qi in q]
    errs = [abs(a - b) for a, b in zip(lut, deq)]
    return QuantResult(
        scale=scale, zero_point=0.0, q=q, lut_deq=deq,
        max_abs_err=max(errs) if errs else 0.0,
        l1_err=sum(errs),
        identity_preserved=False,  # set later after finding the zero-dot slot
        zero_idx=-1,
    )


def quantize_asymmetric(lut: list[float]) -> QuantResult:
    """Asymmetric int8 quant with zero-point shift.

    Maps [lut_min, lut_max] -> [-128, 127] with
      scale = (lut_max - lut_min) / 255
      zero_point = round(-lut_min/scale - 128)
      q = round(v/scale + zero_point), clipped
    """
    lo = min(lut) if lut else 0.0
    hi = max(lut) if lut else 0.0
    if hi == lo:
        # constant LUT (often zero LUT)
        q = [0] * len(lut)
        deq = [float(lo)] * len(lut)
        return QuantResult(
            scale=1.0, zero_point=0.0, q=q, lut_deq=deq,
            max_abs_err=0.0, l1_err=0.0,
            identity_preserved=(lo == 0.0), zero_idx=-1,
        )
    scale = (hi - lo) / 255.0
    # We want qmin->lo, qmax->hi with qmin=-128, qmax=127.
    # v = scale * (q - zp)  =>  q = round(v/scale + zp)
    # at v=lo, q=-128 => -128 = round(lo/scale + zp) => zp = -128 - lo/scale
    zp = -128.0 - lo / scale
    q = []
    for v in lut:
        qi = int(round(v / scale + zp))
        if qi > 127:
            qi = 127
        elif qi < -128:
            qi = -128
        q.append(qi)
    deq = [scale * (qi - zp) for qi in q]
    errs = [abs(a - b) for a, b in zip(lut, deq)]
    return QuantResult(
        scale=scale, zero_point=zp, q=q, lut_deq=deq,
        max_abs_err=max(errs) if errs else 0.0,
        l1_err=sum(errs),
        identity_preserved=False,
        zero_idx=-1,
    )


def annotate_identity(qr: QuantResult, lut_min_dot: int) -> None:
    """Flag whether the lut slot corresponding to dot=0 stays exactly zero after quant.

    c19(0,c,rho) == 0 identically for any (c,rho), so the raw float LUT always
    has a 0.0 at the zero-dot slot. We check the dequantized LUT at that slot.
    """
    zero_idx = -lut_min_dot  # dot - lut_min_dot = 0 - min_dot = -min_dot
    if zero_idx < 0 or zero_idx >= len(qr.lut_deq):
        qr.identity_preserved = True  # not applicable (zero dot outside LUT range)
        qr.zero_idx = -1
        return
    qr.zero_idx = zero_idx
    qr.identity_preserved = (qr.lut_deq[zero_idx] == 0.0)


# -----------------------------------------------------------------------------
# Apply per-neuron quantization to a whole network
# -----------------------------------------------------------------------------

def requantize_network(neurons: list[dict], method: str = "absmax", nbits: int = 8) -> tuple[list[dict], list[QuantResult]]:
    """Return (new_neurons, per-neuron QuantResult list) with LUTs replaced by dequant versions."""
    new = []
    results = []
    for n in neurons:
        lut = list(n["lut"])
        if method == "absmax":
            qr = quantize_symmetric_absmax(lut, nbits=nbits)
        elif method == "asymmetric":
            qr = quantize_asymmetric(lut)
        else:
            raise ValueError(f"unknown method {method}")
        annotate_identity(qr, n["lut_min_dot"])
        new_n = dict(n)
        new_n["lut"] = qr.lut_deq
        new.append(new_n)
        results.append(qr)
    return new, results


def quantize_alpha(neurons: list[dict], nbits: int = 8) -> tuple[list[dict], dict]:
    """Quantize the per-neuron alpha values to int symmetric using a shared scale.

    Returns (new_neurons, stats). The shared scale is absmax(|alpha|) / qmax.
    """
    qmax = (1 << (nbits - 1)) - 1
    qmin = -qmax
    absmax = max((abs(float(n["alpha"])) for n in neurons), default=0.0)
    if absmax == 0.0:
        new = []
        for n in neurons:
            nn = dict(n)
            nn["alpha"] = 0.0
            new.append(nn)
        return new, {"scale": 1.0, "max_abs_err": 0.0}
    scale = absmax / qmax
    new = []
    errs = []
    for n in neurons:
        a = float(n["alpha"])
        qi = int(round(a / scale))
        if qi > qmax:
            qi = qmax
        elif qi < qmin:
            qi = qmin
        deq = qi * scale
        errs.append(abs(a - deq))
        nn = dict(n)
        nn["alpha"] = deq
        new.append(nn)
    return new, {"scale": scale, "max_abs_err": max(errs) if errs else 0.0}


def requantize_per_network_absmax(neurons: list[dict]) -> tuple[list[dict], list[QuantResult], float]:
    """Single shared scale across all neurons' LUTs (per-network)."""
    absmax = 0.0
    for n in neurons:
        for v in n["lut"]:
            if abs(v) > absmax:
                absmax = abs(v)
    if absmax == 0.0:
        # all zero
        new = []
        for n in neurons:
            new_n = dict(n)
            new_n["lut"] = [0.0] * len(n["lut"])
            new.append(new_n)
        return new, [], 1.0
    scale = absmax / 127.0
    new = []
    results = []
    for n in neurons:
        lut = list(n["lut"])
        q = []
        for v in lut:
            qi = int(round(v / scale))
            if qi > 127:
                qi = 127
            elif qi < -127:
                qi = -127
            q.append(qi)
        deq = [qi * scale for qi in q]
        errs = [abs(a - b) for a, b in zip(lut, deq)]
        qr = QuantResult(
            scale=scale, zero_point=0.0, q=q, lut_deq=deq,
            max_abs_err=max(errs) if errs else 0.0,
            l1_err=sum(errs),
            identity_preserved=False, zero_idx=-1,
        )
        annotate_identity(qr, n["lut_min_dot"])
        new_n = dict(n)
        new_n["lut"] = deq
        new.append(new_n)
        results.append(qr)
    return new, results, scale


# -----------------------------------------------------------------------------
# Per-neuron effect isolation: quantize one neuron at a time, measure delta
# -----------------------------------------------------------------------------

def per_neuron_effect(neurons_float: list[dict], neurons_i8: list[dict],
                      X: list[list[int]], y: list[int]) -> list[float]:
    """For each neuron, measure delta = acc(float everywhere except neuron i -> i8) - acc(float).

    Returns list of per-neuron deltas (negative means accuracy drop).
    """
    base = accuracy_rust(neurons_float, X, y)
    deltas = []
    for i in range(len(neurons_float)):
        mixed = list(neurons_float)
        mixed[i] = neurons_i8[i]
        acc = accuracy_rust(mixed, X, y)
        deltas.append(acc - base)
    return deltas


# -----------------------------------------------------------------------------
# Dataset loaders
# -----------------------------------------------------------------------------

def load_seed_final_json(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d


def load_grid3_center_state(path: Path) -> dict:
    return json.loads(path.read_text())


# -----------------------------------------------------------------------------
# Parity check: verify our Rust-semantics Python predict matches the
# ensemble_train/val/test numbers the Rust grower logged.
# -----------------------------------------------------------------------------

def parity_check(name: str, task: str, neurons: list[dict], logged: dict) -> dict:
    tr_x, tr_y, va_x, va_y, te_x, te_y = gen_data(task)
    tr_acc = accuracy_rust(neurons, tr_x, tr_y)
    va_acc = accuracy_rust(neurons, va_x, va_y)
    te_acc = accuracy_rust(neurons, te_x, te_y)
    lg_tr = logged.get("ensemble_train")
    lg_va = logged.get("ensemble_val")
    lg_te = logged.get("ensemble_test")
    return {
        "name": name,
        "task": task,
        "n_neurons": len(neurons),
        "py_train": tr_acc,
        "py_val": va_acc,
        "py_test": te_acc,
        "rs_train": lg_tr,
        "rs_val": lg_va,
        "rs_test": lg_te,
        "match": (
            abs((tr_acc - (lg_tr or 0)) or 0) < 0.01
            and abs((va_acc - (lg_va or 0)) or 0) < 0.01
            and abs((te_acc - (lg_te or 0)) or 0) < 0.01
        ),
    }


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------

def lut_stats(lut: list[float]) -> dict:
    if not lut:
        return {"n": 0, "abs_max": 0.0, "abs_mean": 0.0}
    absvals = [abs(v) for v in lut]
    return {
        "n": len(lut),
        "abs_max": max(absvals),
        "abs_mean": sum(absvals) / len(absvals),
        "min": min(lut),
        "max": max(lut),
    }


def fmt_float(x: float, w: int = 7, p: int = 3) -> str:
    return f"{x:>{w}.{p}f}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def evaluate_network(name: str, task: str, neurons: list[dict], logged: dict | None,
                     verbose: bool = False) -> dict:
    tr_x, tr_y, va_x, va_y, te_x, te_y = gen_data(task)
    fl_tr = accuracy_rust(neurons, tr_x, tr_y)
    fl_va = accuracy_rust(neurons, va_x, va_y)
    fl_te = accuracy_rust(neurons, te_x, te_y)

    # Decision boundary margin analysis on val set
    val_scores_float = [score_rust(neurons, x) for x in va_x]
    abs_margins = sorted([abs(s) for s in val_scores_float])
    min_margin = abs_margins[0] if abs_margins else 0.0
    p10_margin = abs_margins[len(abs_margins)//10] if abs_margins else 0.0
    median_margin = abs_margins[len(abs_margins)//2] if abs_margins else 0.0
    margin_stats = {"min": min_margin, "p10": p10_margin, "median": median_margin,
                    "n_under_0p1": sum(1 for m in abs_margins if m < 0.1)}

    n_absmax, q_absmax = requantize_network(neurons, method="absmax", nbits=8)
    # Actual empirical score shift on val set (not worst-case bound)
    val_scores_i8 = [score_rust(n_absmax, x) for x in va_x]
    score_shifts = [abs(a - b) for a, b in zip(val_scores_float, val_scores_i8)]
    empirical_max_shift = max(score_shifts, default=0.0)
    empirical_median_shift = sorted(score_shifts)[len(score_shifts)//2] if score_shifts else 0.0
    # How close does the i8 ensemble get to flipping any sample?
    # We want min(abs(i8_score)) - that's the closest post-quant margin
    i8_min_margin = min((abs(s) for s in val_scores_i8), default=0.0)
    n_asym, q_asym = requantize_network(neurons, method="asymmetric")
    n_pn_absmax, q_pn_absmax, pn_scale = requantize_per_network_absmax(neurons)
    # Stress test: int4 to prove pipeline sensitivity
    n_i4, q_i4 = requantize_network(neurons, method="absmax", nbits=4)

    def acc_triple(nn):
        return (
            accuracy_rust(nn, tr_x, tr_y),
            accuracy_rust(nn, va_x, va_y),
            accuracy_rust(nn, te_x, te_y),
        )

    ab_tr, ab_va, ab_te = acc_triple(n_absmax)
    as_tr, as_va, as_te = acc_triple(n_asym)
    pn_tr, pn_va, pn_te = acc_triple(n_pn_absmax)
    i4_tr, i4_va, i4_te = acc_triple(n_i4)

    # Combined: int8 LUT + int8 alpha
    n_lut_i8_then_alpha_i8, alpha_q8_stats = quantize_alpha(n_absmax, nbits=8)
    la8_tr, la8_va, la8_te = acc_triple(n_lut_i8_then_alpha_i8)
    # Combined: int8 LUT + int16 alpha
    n_lut_i8_then_alpha_i16, alpha_q16_stats = quantize_alpha(n_absmax, nbits=16)
    la16_tr, la16_va, la16_te = acc_triple(n_lut_i8_then_alpha_i16)

    # Per-neuron isolation (absmax only; that's the recommended path)
    per_nd_va = per_neuron_effect(neurons, n_absmax, va_x, va_y)
    per_nd_te = per_neuron_effect(neurons, n_absmax, te_x, te_y)

    # Worst-case effective score contribution error:
    #   eff_err_i = alpha_i * max_abs_err_i
    # This is the max the ensemble score can shift PER NEURON, so the
    # upper bound on the full ensemble score shift is sum over all neurons.
    # If this sum is smaller than the smallest observed decision margin,
    # then int8 quant cannot flip the classifier prediction.
    eff_errs = [float(n["alpha"]) * qr.max_abs_err for n, qr in zip(neurons, q_absmax)]
    worst_score_shift = sum(eff_errs)
    largest_single = max(eff_errs, default=0.0)

    # LUT stats
    lut_info = [lut_stats(n["lut"]) for n in neurons]

    # Memory footprint
    float_bytes = sum(len(n["lut"]) * 4 for n in neurons)
    i8_bytes = sum(len(n["lut"]) + 4 for n in neurons)  # lut bytes + 4-byte scale per neuron
    pn_bytes = sum(len(n["lut"]) for n in neurons) + 4  # one global scale

    worst_abs_err = max((qr.max_abs_err for qr in q_absmax), default=0.0)
    sum_l1 = sum(qr.l1_err for qr in q_absmax)
    worst_neuron_idx_abs = max(range(len(q_absmax)), key=lambda i: q_absmax[i].max_abs_err) if q_absmax else -1

    # Identity preservation (c19(0)=0)
    id_preserved_count = sum(1 for qr in q_absmax if qr.identity_preserved)
    id_violations = [(i, qr.lut_deq[qr.zero_idx] if qr.zero_idx >= 0 else None)
                     for i, qr in enumerate(q_absmax) if not qr.identity_preserved]

    result = {
        "name": name,
        "task": task,
        "n_neurons": len(neurons),
        "float": {"train": fl_tr, "val": fl_va, "test": fl_te},
        "logged": logged,
        "absmax": {"train": ab_tr, "val": ab_va, "test": ab_te,
                   "per_neuron_val_delta": per_nd_va,
                   "per_neuron_test_delta": per_nd_te,
                   "worst_abs_err": worst_abs_err,
                   "worst_neuron_idx": worst_neuron_idx_abs,
                   "sum_l1_err": sum_l1,
                   "id_preserved_count": id_preserved_count,
                   "id_violations": id_violations,
                   "eff_errs": eff_errs,
                   "worst_score_shift": worst_score_shift,
                   "largest_single_score_err": largest_single,
                   "per_neuron_q": q_absmax},
        "asymmetric": {"train": as_tr, "val": as_va, "test": as_te,
                       "worst_abs_err": max((qr.max_abs_err for qr in q_asym), default=0.0),
                       "sum_l1_err": sum(qr.l1_err for qr in q_asym),
                       "id_preserved_count": sum(1 for qr in q_asym if qr.identity_preserved),
                       "per_neuron_q": q_asym},
        "per_network_absmax": {"train": pn_tr, "val": pn_va, "test": pn_te,
                               "global_scale": pn_scale,
                               "worst_abs_err": max((qr.max_abs_err for qr in q_pn_absmax), default=0.0),
                               "id_preserved_count": sum(1 for qr in q_pn_absmax if qr.identity_preserved),
                               "per_neuron_q": q_pn_absmax},
        "int4_absmax": {"train": i4_tr, "val": i4_va, "test": i4_te,
                        "worst_abs_err": max((qr.max_abs_err for qr in q_i4), default=0.0),
                        "id_preserved_count": sum(1 for qr in q_i4 if qr.identity_preserved),
                        "per_neuron_q": q_i4},
        "lut_i8_alpha_i8": {"train": la8_tr, "val": la8_va, "test": la8_te,
                            "alpha_stats": alpha_q8_stats},
        "lut_i8_alpha_i16": {"train": la16_tr, "val": la16_va, "test": la16_te,
                             "alpha_stats": alpha_q16_stats},
        "margin_stats": margin_stats,
        "empirical_shift": {"max": empirical_max_shift, "median": empirical_median_shift,
                            "i8_min_margin": i8_min_margin},
        "lut_stats": lut_info,
        "footprint": {"float_bytes": float_bytes, "i8_bytes": i8_bytes, "pn_bytes": pn_bytes},
    }
    return result


def print_network(r: dict, verbose: bool = False) -> None:
    print(f"\n=== {r['name']}  [{r['task']}]  neurons={r['n_neurons']} ===")
    if r["logged"]:
        print(f"  logged   : train={r['logged'].get('ensemble_train',0):.2f}  val={r['logged'].get('ensemble_val',0):.2f}  test={r['logged'].get('ensemble_test',0):.2f}")
    f = r["float"]
    a = r["absmax"]
    ay = r["asymmetric"]
    p = r["per_network_absmax"]
    i4 = r["int4_absmax"]
    print(f"  py-float : train={f['train']:.2f}  val={f['val']:.2f}  test={f['test']:.2f}")
    print(f"  i8 absmax: train={a['train']:.2f}  val={a['val']:.2f}  test={a['test']:.2f}  "
          f"(dv={a['val']-f['val']:+.2f} dt={a['test']-f['test']:+.2f})")
    print(f"  i8 asym  : train={ay['train']:.2f}  val={ay['val']:.2f}  test={ay['test']:.2f}  "
          f"(dv={ay['val']-f['val']:+.2f} dt={ay['test']-f['test']:+.2f})")
    print(f"  i8 per-net absmax: train={p['train']:.2f}  val={p['val']:.2f}  test={p['test']:.2f}  "
          f"(dv={p['val']-f['val']:+.2f} dt={p['test']-f['test']:+.2f})")
    print(f"  i4 absmax STRESS: train={i4['train']:.2f}  val={i4['val']:.2f}  test={i4['test']:.2f}  "
          f"(dv={i4['val']-f['val']:+.2f} dt={i4['test']-f['test']:+.2f}, worst_err={i4['worst_abs_err']:.4f})")
    la8 = r["lut_i8_alpha_i8"]
    la16 = r["lut_i8_alpha_i16"]
    print(f"  i8 LUT + i8 alpha : val={la8['val']:.2f}  test={la8['test']:.2f}  "
          f"(dv={la8['val']-f['val']:+.2f} dt={la8['test']-f['test']:+.2f}, alpha_err={la8['alpha_stats']['max_abs_err']:.6f})")
    print(f"  i8 LUT + i16 alpha: val={la16['val']:.2f}  test={la16['test']:.2f}  "
          f"(dv={la16['val']-f['val']:+.2f} dt={la16['test']-f['test']:+.2f}, alpha_err={la16['alpha_stats']['max_abs_err']:.8f})")
    print(f"  worst abs_err (absmax) : {a['worst_abs_err']:.6f} at neuron {a['worst_neuron_idx']}")
    print(f"  sum L1 err (absmax)    : {a['sum_l1_err']:.6f}")
    print(f"  worst score shift (sum alpha*max_err): {a['worst_score_shift']:.6f}  (largest single={a['largest_single_score_err']:.6f})")
    print(f"  c19(0)=0 preserved     : {a['id_preserved_count']}/{r['n_neurons']} neurons (absmax)")
    if a["id_violations"]:
        print(f"    id violations: {[(i, v) for i, v in a['id_violations']]}")
    print(f"  footprint: float={r['footprint']['float_bytes']}B  i8+scale={r['footprint']['i8_bytes']}B  per-net={r['footprint']['pn_bytes']}B")
    ms = r["margin_stats"]
    print(f"  val margin: min={ms['min']:.4f}  p10={ms['p10']:.4f}  median={ms['median']:.4f}  n<0.1={ms['n_under_0p1']}")
    es = r["empirical_shift"]
    print(f"  empirical i8 shift: max={es['max']:.6f}  median={es['median']:.6f}  i8_min_margin={es['i8_min_margin']:.4f}")

    if verbose:
        print(f"\n  per-neuron table (absmax):")
        print(f"    {'n':>3} {'lut_sz':>6} {'abs_max':>8} {'scale':>10} {'max_err':>9} {'dv':>7} {'dt':>7} {'id':>4}")
        for i, qr in enumerate(a["per_neuron_q"]):
            ls = r["lut_stats"][i]
            idok = "ok" if qr.identity_preserved else "OFF"
            print(f"    {i:>3} {ls['n']:>6} {ls['abs_max']:>8.4f} {qr.scale:>10.6f} "
                  f"{qr.max_abs_err:>9.6f} {a['per_neuron_val_delta'][i]:>+7.2f} "
                  f"{a['per_neuron_test_delta'][i]:>+7.2f} {idok:>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--json", type=str, default=None, help="dump all results to this file")
    args = ap.parse_args()

    print("c19 int8 quant analysis\n" + "="*40)

    results = []

    # 1) grid3_center manual smoke
    grid3_center_state = load_grid3_center_state(ROOT / "target" / "c19_manual_grow" / "grid3_center" / "state.json")
    r1 = evaluate_network(
        "grid3_center_manual",
        "grid3_center",
        grid3_center_state["neurons"],
        logged=None,
        verbose=args.verbose,
    )
    print_network(r1, verbose=args.verbose)
    results.append(r1)

    # 2) seed_04 best (89.75 val, 93.25 test)
    seed04 = load_seed_final_json(ROOT / "target" / "c19_parity_sweep" / "seed_04" / "final.json")
    r2 = evaluate_network(
        "parity_sweep_seed_04_best",
        "grid3_full_parity",
        seed04["neurons"],
        logged={"ensemble_train": seed04["ensemble_train"],
                "ensemble_val":   seed04["ensemble_val"],
                "ensemble_test":  seed04["ensemble_test"]},
        verbose=args.verbose,
    )
    print_network(r2, verbose=args.verbose)
    results.append(r2)

    # 3) seed_09 mid-range (78.25 val, 80.25 test)
    seed09 = load_seed_final_json(ROOT / "target" / "c19_parity_sweep" / "seed_09" / "final.json")
    r3 = evaluate_network(
        "parity_sweep_seed_09_mid",
        "grid3_full_parity",
        seed09["neurons"],
        logged={"ensemble_train": seed09["ensemble_train"],
                "ensemble_val":   seed09["ensemble_val"],
                "ensemble_test":  seed09["ensemble_test"]},
        verbose=args.verbose,
    )
    print_network(r3, verbose=args.verbose)
    results.append(r3)

    # 4) smoke_parity 21 neurons
    smoke = load_seed_final_json(ROOT / "target" / "c19_grower_smoke_parity" / "final.json")
    r4 = evaluate_network(
        "smoke_parity_21n",
        "grid3_full_parity",
        smoke["neurons"],
        logged={"ensemble_train": smoke["ensemble_train"],
                "ensemble_val":   smoke["ensemble_val"],
                "ensemble_test":  smoke["ensemble_test"]},
        verbose=args.verbose,
    )
    print_network(r4, verbose=args.verbose)
    results.append(r4)

    # Summary
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"{'network':<32} {'N':>3} {'fl_val':>7} {'i8_val':>7} {'dv':>6} {'fl_test':>8} {'i8_test':>7} {'dt':>6}  {'worst_err':>9}")
    for r in results:
        f = r["float"]
        a = r["absmax"]
        print(f"{r['name']:<32} {r['n_neurons']:>3} "
              f"{f['val']:>7.2f} {a['val']:>7.2f} {a['val']-f['val']:>+6.2f} "
              f"{f['test']:>8.2f} {a['test']:>7.2f} {a['test']-f['test']:>+6.2f}  "
              f"{a['worst_abs_err']:>9.5f}")

    if args.json:
        # strip QuantResult dataclasses for JSON
        def q2dict(qr):
            return {
                "scale": qr.scale,
                "zero_point": qr.zero_point,
                "q": qr.q,
                "lut_deq": qr.lut_deq,
                "max_abs_err": qr.max_abs_err,
                "l1_err": qr.l1_err,
                "identity_preserved": qr.identity_preserved,
                "zero_idx": qr.zero_idx,
            }
        for r in results:
            for key in ("absmax", "asymmetric", "per_network_absmax"):
                if "per_neuron_q" in r[key]:
                    r[key]["per_neuron_q"] = [q2dict(qr) for qr in r[key]["per_neuron_q"]]
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\n[json] wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
