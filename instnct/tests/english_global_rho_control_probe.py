"""
INSTNCT English Global Rho Control Probe
=======================================
Temporary side probe for testing whether a single global rho scalar should
stay fixed, be learned externally, or become part of the network's own active
 prediction loop.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes


IO = 256
NV = 4
TICKS = 8
INPUT_TICKS = 2
TRAIN_SEQS = 2
EVAL_SEQS = 10
SEQ_LEN = 200
WORKERS = 18
BUDGET = 1000
REPORT_EVERY = 50
THRESHOLD = 0.00005
PROJECTION_SCALE = 1.0
THETA_INIT = 0.0
DECAY_INIT_LO = 0.08
DECAY_INIT_HI = 0.24
INITIAL_RHO = 0.3
CONTROL_SEED = 20260327
CONTROL_INPUT_SCALE = 1.0
CONTROL_OUTPUT_SCALE_FRACTION = 0.15
SATURATION_EPS = 0.05
SCHEDULE = ["add", "add", "flip", "decay", "decay", "decay", "decay", "rho"]
MODE_ORDER = ("fixed_global_rho", "learnable_global_rho", "prewired_global_rho")
MODE_LABELS = {
    "fixed_global_rho": "Fixed global rho=0.3",
    "learnable_global_rho": "Externally learnable global rho",
    "prewired_global_rho": "Prewired active global rho loop",
}

_bp = None
_all_data = None
_seq_len = None
_n_train = None
_input_projection = None
_output_projection = None
_bigram = None
_polarity = None
_freq_g = None
_phase_g = None
_ticks_g = None
_input_ticks_g = None
_mode_g = None
_rho_in_proj = None
_rho_out_proj = None
_rho_out_scale = None


def make_bp(io_dim: int, seed: int = 12345) -> np.ndarray:
    rng = np.random.RandomState(seed)
    proj = rng.randn(256, io_dim).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    return proj


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def make_control_vectors(hidden_size: int, seed: int = CONTROL_SEED) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.RandomState(seed)
    rho_in = rng.randn(hidden_size).astype(np.float32)
    rho_out = rng.randn(hidden_size).astype(np.float32)
    rho_in /= np.linalg.norm(rho_in) + 1e-8
    rho_out /= np.linalg.norm(rho_out) + 1e-8
    rho_in *= np.float32(CONTROL_INPUT_SCALE)
    rho_out_scale = float(max(1.0, SelfWiringGraph.MAX_CHARGE * np.sqrt(hidden_size) * CONTROL_OUTPUT_SCALE_FRACTION))
    return rho_in, rho_out, rho_out_scale


def init_worker(bp, all_data, seq_len, n_train, wi, wo, bg, polarity, freq, phase, ticks, input_ticks, mode, rho_in, rho_out, rho_out_scale):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram
    global _polarity, _freq_g, _phase_g, _ticks_g, _input_ticks_g, _mode_g
    global _rho_in_proj, _rho_out_proj, _rho_out_scale
    _bp = bp
    _all_data = all_data
    _seq_len = seq_len
    _n_train = n_train
    _input_projection = wi
    _output_projection = wo
    _bigram = bg
    _polarity = polarity
    _freq_g = freq
    _phase_g = phase
    _ticks_g = ticks
    _input_ticks_g = input_ticks
    _mode_g = mode
    _rho_in_proj = rho_in
    _rho_out_proj = rho_out
    _rho_out_scale = rho_out_scale


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    np.nan_to_num(scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    denom = float(np.sum(exp_scores))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full(scores.shape, 1.0 / len(scores), dtype=np.float32)
    return exp_scores / denom


def build_sequences(all_data: np.ndarray, seq_len: int, n_eval: int) -> list[np.ndarray]:
    eval_rng = np.random.RandomState(9999)
    return [
        all_data[offset : offset + seq_len]
        for offset in [int(eval_rng.randint(0, len(all_data) - seq_len)) for _ in range(n_eval)]
    ]


def append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def dump_json(json_path: Path, payload: dict) -> None:
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_initial_state(io_dim, hidden_ratio, projection_scale, theta_init, decay_lo, decay_hi):
    ref = SelfWiringGraph(io_dim, hidden_ratio=hidden_ratio, projection_scale=projection_scale, seed=42)
    h_size = ref.H
    decay_rng = np.random.RandomState(99)
    return {
        "H": h_size,
        "mask": np.zeros((h_size, h_size), dtype=np.bool_),
        "theta": np.full(h_size, theta_init, dtype=np.float32),
        "decay": decay_rng.uniform(decay_lo, decay_hi, h_size).astype(np.float32),
        "polarity": ref.polarity.astype(np.float32),
        "freq": ref.freq.astype(np.float32).copy(),
        "phase": ref.phase.astype(np.float32).copy(),
        "input_projection": ref.input_projection.astype(np.float32).copy(),
        "output_projection": ref.output_projection.astype(np.float32).copy(),
    }


def structure_stats(mask: np.ndarray) -> dict[str, int]:
    present = mask
    out_deg = np.sum(present, axis=1)
    in_deg = np.sum(present, axis=0)
    reciprocal_pairs = int(np.triu(present & present.T, k=1).sum())
    sink_count = int(np.sum((in_deg > 0) & (out_deg == 0)))
    source_only_count = int(np.sum((out_deg > 0) & (in_deg == 0)))
    isolated_count = int(np.sum((out_deg == 0) & (in_deg == 0)))
    return {
        "reciprocal_pairs": reciprocal_pairs,
        "sink_count": sink_count,
        "source_only_count": source_only_count,
        "isolated_count": isolated_count,
    }


def global_rho_gate_stats(theta, rho_scalar, freq, phase, ticks):
    tick_ids = np.arange(ticks, dtype=np.float32)[:, None]
    wave = np.sin(tick_ids * freq[None, :] + phase[None, :])
    c_t = np.float32(rho_scalar) * wave
    effective = np.maximum(0.0, theta[None, :] + c_t)
    return {
        "ct_abs_mean": float(np.mean(np.abs(c_t))),
        "ct_min": float(c_t.min()),
        "ct_max": float(c_t.max()),
        "effective_theta_mean": float(np.mean(effective)),
        "effective_theta_min": float(effective.min()),
        "effective_theta_max": float(effective.max()),
    }


def run_rollout(mask, theta, decay, rho_start, text_bytes, mode):
    rs, cs = np.where(mask)
    sp_vals = _polarity[rs]
    ret = 1.0 - decay
    state = np.zeros(mask.shape[0], dtype=np.float32)
    charge = np.zeros(mask.shape[0], dtype=np.float32)
    logits = []
    nonfinite = 0
    rho_curr = float(rho_start)
    rho_values = [rho_curr]
    raw_outputs = []
    sat_low_count = 0
    sat_high_count = 0
    for i in range(len(text_bytes) - 1):
        act = state.copy()
        control_in = None
        if mode == "prewired_global_rho":
            control_in = np.float32(2.0 * rho_curr - 1.0) * _rho_in_proj
        for tick in range(_ticks_g):
            if tick < _input_ticks_g:
                act = act + (_bp[text_bytes[i]] @ _input_projection)
                if control_in is not None:
                    act = act + control_in
            raw = np.zeros(mask.shape[0], dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charge += raw
            np.clip(charge, 0.0, SelfWiringGraph.MAX_CHARGE, out=charge)
            charge *= ret
            wave = np.sin(np.float32(tick) * _freq_g + _phase_g)
            effective_theta = np.maximum(0.0, theta + np.float32(rho_curr) * wave)
            fired = charge >= effective_theta
            act = fired.astype(np.float32) * _polarity
        state = act.copy()
        out = charge @ _output_projection
        if not np.isfinite(out).all():
            nonfinite += 1
            np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        logits.append(out.astype(np.float32, copy=False))
        if mode == "prewired_global_rho":
            raw_scalar = float(charge @ _rho_out_proj)
            if not np.isfinite(raw_scalar):
                nonfinite += 1
                raw_scalar = 0.0
            raw_outputs.append(raw_scalar)
            rho_curr = clamp01(0.5 * (np.tanh(raw_scalar / _rho_out_scale) + 1.0))
        else:
            raw_outputs.append(0.0)
        if rho_curr <= SATURATION_EPS:
            sat_low_count += 1
        if rho_curr >= 1.0 - SATURATION_EPS:
            sat_high_count += 1
        rho_values.append(float(rho_curr))
    return logits, nonfinite, {
        "rho_start": float(rho_values[0]),
        "rho_end": float(rho_values[-1]),
        "rho_values": rho_values,
        "raw_mean": float(np.mean(raw_outputs)) if raw_outputs else 0.0,
        "raw_min": float(np.min(raw_outputs)) if raw_outputs else 0.0,
        "raw_max": float(np.max(raw_outputs)) if raw_outputs else 0.0,
        "sat_low_count": int(sat_low_count),
        "sat_high_count": int(sat_high_count),
    }


def score_bigram(mask, theta, decay, rho_value, seqs, mode):
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    nonfinite = 0
    for text_bytes in seqs:
        logits, bad, _ = run_rollout(mask, theta, decay, rho_value, text_bytes, mode)
        nonfinite += bad
        seq_score = 0.0
        count = 0
        for i, out in enumerate(logits):
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            if not np.isfinite(sims).all():
                nonfinite += 1
                np.nan_to_num(sims, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            pred = softmax(sims)
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += float(cos)
            count += 1
        total += seq_score / count if count else 0.0
    return total / max(len(seqs), 1), nonfinite


def worker_eval(args):
    mask_flat, theta, decay, rho_value, h_size, seed, proposal_type, mode = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(h_size, h_size)
    new_mask = mask
    new_theta = theta
    new_decay = decay
    new_rho = float(rho_value)
    if proposal_type == "add":
        r = rng.randint(0, h_size - 1)
        c = rng.randint(0, h_size - 1)
        if r == c or mask[r, c]:
            return {"delta": -1e9, "type": "add"}
        new_mask = mask.copy()
        new_mask[r, c] = True
    elif proposal_type == "flip":
        alive = list(zip(*np.where(mask)))
        if not alive:
            return {"delta": -1e9, "type": "flip"}
        r, c = alive[rng.randint(0, len(alive) - 1)]
        nc = rng.randint(0, h_size - 1)
        if nc == r or nc == c or mask[r, nc]:
            return {"delta": -1e9, "type": "flip"}
        new_mask = mask.copy()
        new_mask[r, c] = False
        new_mask[r, nc] = True
    elif proposal_type == "decay":
        idx = rng.randint(0, h_size - 1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))
    elif proposal_type == "rho":
        if mode != "learnable_global_rho":
            return {"delta": -1e9, "type": "rho"}
        new_rho = clamp01(rho_value + rng.uniform(-0.1, 0.1))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = int(np_rng.randint(0, data_len - _seq_len))
        seqs.append(_all_data[off : off + _seq_len])
    old_score, _ = score_bigram(mask, theta, decay, rho_value, seqs, mode)
    new_score, _ = score_bigram(new_mask, new_theta, new_decay, new_rho, seqs, mode)
    improved = new_score > old_score
    return {
        "delta": float(new_score - old_score),
        "type": proposal_type,
        "new_mask_flat": new_mask.flatten() if improved and proposal_type in ("add", "flip") else None,
        "new_decay": new_decay if improved and proposal_type == "decay" else None,
        "new_rho": float(new_rho) if improved and proposal_type == "rho" else None,
    }


def eval_accuracy(mask, theta, decay, rho_value, text_bytes, mode):
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    correct = 0
    total = 0
    logits, nonfinite, trace = run_rollout(mask, theta, decay, rho_value, text_bytes, mode)
    for i, out in enumerate(logits):
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if not np.isfinite(sims).all():
            nonfinite += 1
            np.nan_to_num(sims, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if int(np.argmax(sims)) == int(text_bytes[i + 1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0, nonfinite, trace


def evaluate_mode(mask, theta, decay, rho_value, eval_seqs, mode):
    accs = []
    nonfinite = 0
    rho_starts = []
    rho_ends = []
    raw_means = []
    raw_mins = []
    raw_maxs = []
    sat_low_total = 0
    sat_high_total = 0
    predicted_values = []
    for seq in eval_seqs:
        acc, bad, trace = eval_accuracy(mask, theta, decay, rho_value, seq, mode)
        accs.append(acc)
        nonfinite += bad
        rho_starts.append(trace["rho_start"])
        rho_ends.append(trace["rho_end"])
        raw_means.append(trace["raw_mean"])
        raw_mins.append(trace["raw_min"])
        raw_maxs.append(trace["raw_max"])
        sat_low_total += trace["sat_low_count"]
        sat_high_total += trace["sat_high_count"]
        predicted_values.extend(trace["rho_values"][1:])
    return {
        "eval_pct": float(np.mean(accs) * 100.0),
        "nonfinite_events": int(nonfinite),
        "rho_start": float(np.mean(rho_starts)) if rho_starts else float(rho_value),
        "rho_end": float(np.mean(rho_ends)) if rho_ends else float(rho_value),
        "rho_delta": float(np.mean(rho_ends) - np.mean(rho_starts)) if rho_starts else 0.0,
        "controller_raw_mean": float(np.mean(raw_means)) if raw_means else 0.0,
        "controller_raw_min": float(np.min(raw_mins)) if raw_mins else 0.0,
        "controller_raw_max": float(np.max(raw_maxs)) if raw_maxs else 0.0,
        "sat_low_count": int(sat_low_total),
        "sat_high_count": int(sat_high_total),
        "mean_predicted_rho": float(np.mean(predicted_values)) if predicted_values else float(rho_value),
        "predicted_count": int(len(predicted_values)),
    }


def format_report_line(mode_key, step, metrics, edges, accepts, theta, decay, rho_value, struct, gate, elapsed, sps):
    theta_min, theta_max = float(theta.min()), float(theta.max())
    decay_min, decay_max = float(decay.min()), float(decay.max())
    main = (
        f"[{mode_key:>20} {step:5d}] eval={metrics['eval_pct']:5.2f}% edges={edges:4d} "
        f"[Ad={accepts['add']}|Fl={accepts['flip']}|Dc={accepts['decay']}|Rh={accepts['rho']}] "
        f"rho={rho_value:.4f} start={metrics['rho_start']:.4f} end={metrics['rho_end']:.4f} "
        f"delta={metrics['rho_delta']:+.4f} theta={float(np.mean(theta)):.4f}[{theta_min:.3f},{theta_max:.3f}] "
        f"decay={float(np.mean(decay)):.4f}[{decay_min:.3f},{decay_max:.3f}] {elapsed:.0f}s ({sps:.2f} step/s)"
    )
    detail = (
        f"  ctrl raw={metrics['controller_raw_mean']:.4f}[{metrics['controller_raw_min']:.4f},{metrics['controller_raw_max']:.4f}] "
        f"sat0={metrics['sat_low_count']} sat1={metrics['sat_high_count']} pred_rho={metrics['mean_predicted_rho']:.4f} "
        f"gate ct_abs={gate['ct_abs_mean']:.4f} ct=[{gate['ct_min']:.3f},{gate['ct_max']:.3f}] "
        f"effT={gate['effective_theta_mean']:.4f}[{gate['effective_theta_min']:.3f},{gate['effective_theta_max']:.3f}] "
        f"struct recip={struct['reciprocal_pairs']} sink={struct['sink_count']} "
        f"src={struct['source_only_count']} iso={struct['isolated_count']} nonfinite={metrics['nonfinite_events']}"
    )
    report = {
        "step": int(step),
        "eval_pct": round(metrics["eval_pct"], 2),
        "edges": int(edges),
        "accepts": {key: int(value) for key, value in accepts.items()},
        "rho_start": round(metrics["rho_start"], 4),
        "rho_end": round(metrics["rho_end"], 4),
        "rho_delta": round(metrics["rho_delta"], 4),
        "controller_raw_mean": round(metrics["controller_raw_mean"], 4),
        "controller_raw_min": round(metrics["controller_raw_min"], 4),
        "controller_raw_max": round(metrics["controller_raw_max"], 4),
        "sat_low_count": int(metrics["sat_low_count"]),
        "sat_high_count": int(metrics["sat_high_count"]),
        "mean_predicted_rho": round(metrics["mean_predicted_rho"], 4),
        "nonfinite_events": int(metrics["nonfinite_events"]),
        "theta_mean": round(float(np.mean(theta)), 4),
        "theta_min": round(theta_min, 4),
        "theta_max": round(theta_max, 4),
        "decay_mean": round(float(np.mean(decay)), 4),
        "decay_min": round(decay_min, 4),
        "decay_max": round(decay_max, 4),
        "elapsed_sec": int(elapsed),
        "step_per_sec": round(float(sps), 2),
        "structure": struct,
        "gate": gate,
    }
    return main, detail, report


def summarize_mode(mode_key, label, reports, final_eval, final_edges, final_nonfinite, final_struct, final_gate, elapsed, rho_value):
    peak = max(reports, key=lambda item: item["eval_pct"]) if reports else None
    final_report = reports[-1] if reports else {}
    total_predictions = sum(int(item.get("sat_low_count", 0) + item.get("sat_high_count", 0)) for item in reports)
    saturation_rate = 0.0
    predicted_total = sum(int(item.get("predicted_count", 0)) for item in reports if "predicted_count" in item)
    if predicted_total:
        saturation_rate = total_predictions / predicted_total
    rho_move_max = max((abs(float(item.get("rho_delta", 0.0))) for item in reports), default=0.0)
    controller_raw_max = max(
        (
            max(
                abs(float(item.get("controller_raw_min", 0.0))),
                abs(float(item.get("controller_raw_max", 0.0))),
            )
            for item in reports
        ),
        default=0.0,
    )
    return {
        "mode": mode_key,
        "label": label,
        "final_eval_pct": round(final_eval * 100.0, 2),
        "peak_eval_pct": round(float(peak["eval_pct"]), 2) if peak else round(final_eval * 100.0, 2),
        "time_to_peak_sec": int(peak["elapsed_sec"]) if peak else int(elapsed),
        "final_edges": int(final_edges),
        "final_nonfinite_events": int(final_nonfinite),
        "final_rho": round(float(final_report.get("rho_end", rho_value)), 4),
        "final_rho_delta": round(float(final_report.get("rho_delta", 0.0)), 4),
        "mean_predicted_rho": round(float(final_report.get("mean_predicted_rho", rho_value)), 4),
        "saturation_rate": round(float(saturation_rate), 4),
        "max_abs_rho_delta": round(float(rho_move_max), 4),
        "max_abs_controller_raw": round(float(controller_raw_max), 4),
        "final_struct": final_struct,
        "final_gate": final_gate,
        "elapsed_sec": int(elapsed),
        "reports": reports,
    }


def train_mode(mode_key, label, init_state, bp, bigram, all_data, eval_seqs, control, args, live_log, json_log, payload):
    h_size = init_state["H"]
    mask = init_state["mask"].copy()
    theta = init_state["theta"].copy()
    decay = init_state["decay"].copy()
    polarity = init_state["polarity"].copy()
    freq = init_state["freq"].copy()
    phase = init_state["phase"].copy()
    rho_value = float(INITIAL_RHO)
    accepts = {"add": 0, "flip": 0, "decay": 0, "rho": 0}
    reports = []
    t0 = time.time()
    init_worker(bp, all_data, args.seq_len, args.train_seqs, init_state["input_projection"], init_state["output_projection"], bigram, polarity, freq, phase, args.ticks, args.input_ticks, mode_key, control["rho_in"], control["rho_out"], control["rho_out_scale"])
    pool = Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(bp, all_data, args.seq_len, args.train_seqs, init_state["input_projection"], init_state["output_projection"], bigram, polarity, freq, phase, args.ticks, args.input_ticks, mode_key, control["rho_in"], control["rho_out"], control["rho_out_scale"]),
    )
    try:
        for step in range(1, args.budget + 1):
            proposal_type = SCHEDULE[(step - 1) % len(SCHEDULE)]
            if proposal_type == "rho" and mode_key != "learnable_global_rho":
                proposal_type = "decay"
            job_args = [
                (mask.flatten(), theta, decay, rho_value, h_size, 10000 + step * 50 + worker_idx, proposal_type, mode_key)
                for worker_idx in range(args.workers)
            ]
            best = max(pool.map(worker_eval, job_args), key=lambda item: item["delta"])
            if best["delta"] > args.threshold:
                if best.get("new_mask_flat") is not None:
                    mask = best["new_mask_flat"].reshape(h_size, h_size)
                if best.get("new_decay") is not None:
                    decay = best["new_decay"]
                if best.get("new_rho") is not None:
                    rho_value = float(best["new_rho"])
                accepts[best["type"]] += 1
            if step % args.report_every == 0 or step == 1 or step == args.budget:
                elapsed = time.time() - t0
                metrics = evaluate_mode(mask, theta, decay, rho_value, eval_seqs, mode_key)
                edges = int(mask.sum())
                struct = structure_stats(mask)
                gate = global_rho_gate_stats(theta, metrics["rho_end"] if mode_key == "prewired_global_rho" else rho_value, freq, phase, args.ticks)
                line, detail, report = format_report_line(mode_key, step, metrics, edges, accepts, theta, decay, rho_value, struct, gate, elapsed, step / max(elapsed, 1e-8))
                report["predicted_count"] = int(metrics["predicted_count"])
                reports.append(report)
                append_log(live_log, line)
                append_log(live_log, detail)
                print(line)
                print(detail)
                payload["modes"][mode_key] = {
                    "label": label,
                    "reports": reports,
                }
                dump_json(json_log, payload)
                sys.stdout.flush()
    finally:
        pool.terminate()
        pool.join()
    elapsed = time.time() - t0
    metrics = evaluate_mode(mask, theta, decay, rho_value, eval_seqs, mode_key)
    final_struct = structure_stats(mask)
    final_gate = global_rho_gate_stats(theta, metrics["rho_end"] if mode_key == "prewired_global_rho" else rho_value, freq, phase, args.ticks)
    summary = summarize_mode(mode_key, label, reports, metrics["eval_pct"] / 100.0, int(mask.sum()), metrics["nonfinite_events"], final_struct, final_gate, elapsed, rho_value)
    payload["modes"][mode_key]["summary"] = summary
    dump_json(json_log, payload)
    append_log(live_log, f"FINAL {mode_key}: eval={summary['final_eval_pct']:.2f}% edges={summary['final_edges']} rho={summary['final_rho']:.4f} nonfinite={summary['final_nonfinite_events']} elapsed={summary['elapsed_sec']}s")
    print(f"FINAL {mode_key}: eval={summary['final_eval_pct']:.2f}% edges={summary['final_edges']} rho={summary['final_rho']:.4f} nonfinite={summary['final_nonfinite_events']} elapsed={summary['elapsed_sec']}s")
    sys.stdout.flush()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run the English global rho control-loop probe.")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--train-seqs", type=int, default=TRAIN_SEQS)
    parser.add_argument("--eval-seqs", type=int, default=EVAL_SEQS)
    parser.add_argument("--report-every", type=int, default=REPORT_EVERY)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--input-ticks", type=int, default=INPUT_TICKS)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--modes", nargs="*", choices=MODE_ORDER, default=list(MODE_ORDER))
    parser.add_argument("--run-name", default="english_global_rho_control_probe")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    bp = make_bp(IO)
    all_data = load_fineweb_bytes()
    bigram_path = ROOT / "recipes" / "data" / "bigram_table.npy"
    if not bigram_path.exists():
        raise FileNotFoundError(f"Missing bigram table: {bigram_path}")
    bigram = np.load(bigram_path)
    eval_seqs = build_sequences(all_data, args.seq_len, args.eval_seqs)
    init_state = build_initial_state(IO, NV, PROJECTION_SCALE, THETA_INIT, DECAY_INIT_LO, DECAY_INIT_HI)
    rho_in, rho_out, rho_out_scale = make_control_vectors(init_state["H"], seed=CONTROL_SEED)
    control = {
        "rho_in": rho_in,
        "rho_out": rho_out,
        "rho_out_scale": rho_out_scale,
    }
    run_prefix = ROOT / "tests" / args.run_name
    live_log = run_prefix.with_name(f"{args.run_name}_live.txt")
    json_log = run_prefix.with_name(f"{args.run_name}_results.json")
    payload = {
        "config": {
            "budget": args.budget,
            "workers": args.workers,
            "seq_len": args.seq_len,
            "train_seqs": args.train_seqs,
            "eval_seqs": args.eval_seqs,
            "ticks": args.ticks,
            "input_ticks": args.input_ticks,
            "threshold": args.threshold,
            "modes": list(args.modes),
            "initial_rho": INITIAL_RHO,
            "control_seed": CONTROL_SEED,
            "rho_out_scale": rho_out_scale,
        },
        "modes": {},
        "summary_table": [],
        "promotion_verdict": {},
    }
    live_log.write_text("", encoding="utf-8")
    append_log(live_log, f"=== START {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    append_log(live_log, f"Loaded {len(all_data) / 1e6:.1f} MB text")
    append_log(live_log, f"Bigram table: {tuple(bigram.shape)}")
    append_log(live_log, f"Modes: {', '.join(args.modes)}")
    append_log(live_log, f"Budget={args.budget} workers={args.workers} seq_len={args.seq_len} threshold={args.threshold}")
    dump_json(json_log, payload)
    print(f"Loaded {len(all_data) / 1e6:.1f} MB text")
    print(f"Bigram table: {tuple(bigram.shape)}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Budget={args.budget} workers={args.workers} seq_len={args.seq_len} threshold={args.threshold}")
    print(f"Live log: {live_log}")
    print(f"JSON log: {json_log}")
    sys.stdout.flush()
    summaries = []
    for mode_key in args.modes:
        summary = train_mode(mode_key, MODE_LABELS[mode_key], init_state, bp, bigram, all_data, eval_seqs, control, args, live_log, json_log, payload)
        summaries.append(summary)
    baseline = next(item for item in summaries if item["mode"] == "fixed_global_rho")
    external = next((item for item in summaries if item["mode"] == "learnable_global_rho"), None)
    prewired = next((item for item in summaries if item["mode"] == "prewired_global_rho"), None)
    prewired_alive = bool(
        prewired
        and prewired["max_abs_rho_delta"] > 0.01
        and prewired["max_abs_controller_raw"] > 0.01
    )
    prewired_not_saturated = bool(prewired and prewired["saturation_rate"] < 0.95)
    prewired_matches_baseline = bool(prewired and prewired["final_eval_pct"] >= baseline["final_eval_pct"])
    prewired_vs_external = None if external is None or prewired is None else round(prewired["final_eval_pct"] - external["final_eval_pct"], 2)
    promotion = {
        "baseline_mode": baseline["mode"],
        "baseline_final_eval_pct": baseline["final_eval_pct"],
        "external_final_eval_pct": external["final_eval_pct"] if external else None,
        "prewired_final_eval_pct": prewired["final_eval_pct"] if prewired else None,
        "prewired_alive": prewired_alive,
        "prewired_not_saturated": prewired_not_saturated,
        "prewired_matches_baseline": prewired_matches_baseline,
        "prewired_minus_external_eval_pct": prewired_vs_external,
        "promote": bool(prewired and prewired_alive and prewired_not_saturated and prewired_matches_baseline and prewired["final_nonfinite_events"] == 0),
    }
    promotion["verdict"] = "PROMOTE prewired_global_rho" if promotion["promote"] else "KEEP CANONICAL MAINLINE"
    payload["summary_table"] = summaries
    payload["promotion_verdict"] = promotion
    dump_json(json_log, payload)
    print("\n" + "=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)
    for item in summaries:
        struct = item["final_struct"]
        print(
            f"{item['mode']:>20} final={item['final_eval_pct']:6.2f}% peak={item['peak_eval_pct']:6.2f}% "
            f"t_peak={item['time_to_peak_sec']:5d}s edges={item['final_edges']:4d} rho={item['final_rho']:.4f} "
            f"pred={item['mean_predicted_rho']:.4f} sat={item['saturation_rate']:.2f} "
            f"recip={struct['reciprocal_pairs']:3d} sink={struct['sink_count']:4d} src={struct['source_only_count']:4d} iso={struct['isolated_count']:4d}"
        )
    print("-" * 76)
    print(
        f"Promotion verdict: {promotion['verdict']} "
        f"(baseline={baseline['final_eval_pct']:.2f}% prewired={promotion['prewired_final_eval_pct']})"
    )
    print(f"Live log saved: {live_log}")
    print(f"JSON report saved: {json_log}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
