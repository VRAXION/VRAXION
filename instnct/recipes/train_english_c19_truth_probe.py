"""
INSTNCT English C19 Truth Probe
================================
Temporary side recipe for proving the next C19 change before any bake into
the canonical English lane.

Modes:
  - fixed_additive: current canonical recipe semantics with fixed rho/freq/phase
  - additive_rho:   current recipe semantics with learnable rho
  - graph_exact_rho: exact graph.py rollout semantics with learnable rho

This script keeps the canonical English recipe untouched and emits both a
human-readable live log and a machine-readable JSON report.
"""
import argparse
import json
import os
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
BASELINE_SCHEDULE = ["add", "add", "flip", "decay", "decay", "decay", "decay", "decay"]
LEARNABLE_RHO_SCHEDULE = ["add", "add", "flip", "decay", "decay", "decay", "decay", "rho"]
MODE_ORDER = ("fixed_additive", "additive_rho", "graph_exact_rho")
MODE_LABELS = {
    "fixed_additive": "Baseline fixed additive C19",
    "additive_rho": "Learnable rho with additive recipe C19",
    "graph_exact_rho": "Learnable rho with exact graph.py C19",
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


def make_bp(io_dim: int, seed: int = 12345) -> np.ndarray:
    rng = np.random.RandomState(seed)
    proj = rng.randn(256, io_dim).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    return proj


def init_worker(bp, all_data, seq_len, n_train, wi, wo, bg, polarity, freq, phase, ticks, input_ticks, mode):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram
    global _polarity, _freq_g, _phase_g, _ticks_g, _input_ticks_g, _mode_g
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


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    np.nan_to_num(scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    denom = float(np.sum(exp_scores))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full(scores.shape, 1.0 / len(scores), dtype=np.float32)
    return exp_scores / denom


def run_recipe_rollout(mask, theta, decay, rho, text_bytes):
    """Current canonical recipe semantics."""
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs] * _polarity[rs]
    ret = 1.0 - decay
    state = np.zeros(mask.shape[0], dtype=np.float32)
    charge = np.zeros(mask.shape[0], dtype=np.float32)
    logits = []
    nonfinite = 0
    for i in range(len(text_bytes) - 1):
        act = state.copy()
        for tick in range(_ticks_g):
            if tick < _input_ticks_g:
                act = act + (_bp[text_bytes[i]] @ _input_projection)
            raw = np.zeros(mask.shape[0], dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charge += raw
            np.clip(charge, 0.0, SelfWiringGraph.MAX_CHARGE, out=charge)
            charge *= ret
            wave = np.sin(np.float32(tick) * _freq_g + _phase_g)
            effective_theta = np.maximum(0.0, theta + rho * wave)
            fired = charge >= effective_theta
            act = fired.astype(np.float32) * _polarity
        state = act.copy()
        out = charge @ _output_projection
        if not np.isfinite(out).all():
            nonfinite += 1
            np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        logits.append(out.astype(np.float32, copy=False))
    return logits, nonfinite


def run_graph_rollout(mask, theta, decay, rho, text_bytes):
    """Exact graph.py rollout semantics."""
    state = np.zeros(mask.shape[0], dtype=np.float32)
    charge = np.zeros(mask.shape[0], dtype=np.float32)
    refractory = np.zeros(mask.shape[0], dtype=np.int8)
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    logits = []
    nonfinite = 0
    for i in range(len(text_bytes) - 1):
        injected = (_bp[text_bytes[i]] @ _input_projection).astype(np.float32, copy=False)
        state, charge = SelfWiringGraph.rollout_token(
            injected,
            mask=mask,
            theta=theta,
            decay=decay,
            ticks=_ticks_g,
            input_duration=_input_ticks_g,
            state=state,
            charge=charge,
            sparse_cache=sparse_cache,
            edge_magnitude=1.0,
            polarity=_polarity,
            refractory=refractory,
            freq=_freq_g,
            phase=_phase_g,
            rho=rho,
        )
        out = state @ _output_projection
        if not np.isfinite(out).all():
            nonfinite += 1
            np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        logits.append(out.astype(np.float32, copy=False))
    return logits, nonfinite


def score_bigram(mask, theta, decay, rho, seqs, mode):
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    nonfinite = 0
    for text_bytes in seqs:
        if mode == "graph_exact_rho":
            logits, bad = run_graph_rollout(mask, theta, decay, rho, text_bytes)
        else:
            logits, bad = run_recipe_rollout(mask, theta, decay, rho, text_bytes)
        nonfinite += bad
        seq_score = 0.0
        n = 0
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
            n += 1
        total += seq_score / n if n else 0.0
    return total / max(len(seqs), 1), nonfinite


def eval_accuracy(mask, theta, decay, rho, text_bytes, bp, input_projection, output_projection, polarity, freq, phase, mode):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    correct = 0
    total = 0
    if mode == "graph_exact_rho":
        logits, nonfinite = run_graph_rollout(mask, theta, decay, rho, text_bytes)
    else:
        logits, nonfinite = run_recipe_rollout(mask, theta, decay, rho, text_bytes)
    for i, out in enumerate(logits):
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if not np.isfinite(sims).all():
            nonfinite += 1
            np.nan_to_num(sims, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if int(np.argmax(sims)) == int(text_bytes[i + 1]):
            correct += 1
        total += 1
    acc = correct / total if total else 0.0
    return acc, nonfinite


def worker_eval(args):
    mask_flat, theta, decay, rho, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask
    new_theta = theta
    new_decay = decay
    new_rho = rho

    if proposal_type == "add":
        r = rng.randint(0, H - 1)
        c = rng.randint(0, H - 1)
        if r == c or mask[r, c] != 0:
            return {"delta": -1e9, "type": "add"}
        new_mask = mask.copy()
        new_mask[r, c] = 1.0
    elif proposal_type == "flip":
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {"delta": -1e9, "type": "flip"}
        r, c = alive[rng.randint(0, len(alive) - 1)]
        nc = rng.randint(0, H - 1)
        if nc == r or nc == c or mask[r, nc] != 0:
            return {"delta": -1e9, "type": "flip"}
        new_mask = mask.copy()
        new_mask[r, c] = 0.0
        new_mask[r, nc] = 1.0
    elif proposal_type == "decay":
        idx = rng.randint(0, H - 1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))
    elif proposal_type == "rho":
        idx = rng.randint(0, H - 1)
        new_rho = rho.copy()
        new_rho[idx] = max(0.0, min(1.0, rho[idx] + rng.uniform(-0.1, 0.1)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = int(np_rng.randint(0, data_len - _seq_len))
        seqs.append(_all_data[off : off + _seq_len])

    old_score, _ = score_bigram(mask, theta, decay, rho, seqs, _mode_g)
    new_score, _ = score_bigram(new_mask, new_theta, new_decay, new_rho, seqs, _mode_g)
    improved = new_score > old_score
    return {
        "delta": float(new_score - old_score),
        "type": proposal_type,
        "new_mask_flat": new_mask.flatten() if improved and proposal_type in ("add", "flip") else None,
        "new_theta": new_theta if improved and proposal_type == "theta" else None,
        "new_decay": new_decay if improved and proposal_type == "decay" else None,
        "new_rho": new_rho if improved and proposal_type == "rho" else None,
    }


def structure_stats(mask: np.ndarray) -> dict[str, int]:
    present = mask != 0
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


def additive_gate_stats(theta, rho, freq, phase, ticks):
    tick_ids = np.arange(ticks, dtype=np.float32)[:, None]
    wave = np.sin(tick_ids * freq[None, :] + phase[None, :])
    c_t = rho[None, :] * wave
    effective = np.maximum(0.0, theta[None, :] + c_t)
    return {
        "gate_kind": "additive_ct",
        "ct_abs_mean": float(np.mean(np.abs(c_t))),
        "ct_min": float(c_t.min()),
        "ct_max": float(c_t.max()),
        "effective_theta_mean": float(np.mean(effective)),
        "effective_theta_min": float(effective.min()),
        "effective_theta_max": float(effective.max()),
    }


def graph_gate_stats(theta, rho, freq, phase, ticks):
    tick_ids = np.arange(ticks, dtype=np.float32)[:, None]
    wave = np.sin(tick_ids * freq[None, :] + phase[None, :])
    gate = 1.0 + rho[None, :] * wave
    effective = np.clip(theta[None, :] * gate, 1.0, SelfWiringGraph.MAX_CHARGE)
    return {
        "gate_kind": "graph_factor",
        "gate_mean": float(np.mean(gate)),
        "gate_min": float(gate.min()),
        "gate_max": float(gate.max()),
        "effective_theta_mean": float(np.mean(effective)),
        "effective_theta_min": float(effective.min()),
        "effective_theta_max": float(effective.max()),
    }


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


def summarize_mode(mode_key, label, reports, init_rho, final_eval, final_edges, final_nonfinite, final_struct, final_gate, elapsed):
    peak = max(reports, key=lambda item: item["eval_pct"]) if reports else None
    final_report = reports[-1] if reports else {}
    return {
        "mode": mode_key,
        "label": label,
        "final_eval_pct": round(final_eval * 100.0, 2),
        "peak_eval_pct": round(float(peak["eval_pct"]), 2) if peak else round(final_eval * 100.0, 2),
        "time_to_peak_sec": int(peak["elapsed_sec"]) if peak else int(elapsed),
        "final_edges": int(final_edges),
        "final_nonfinite_events": int(final_nonfinite),
        "final_rho_mean": round(float(final_report.get("rho_mean", float(np.mean(init_rho)))), 4),
        "final_rho_min": round(float(final_report.get("rho_min", float(np.min(init_rho)))), 4),
        "final_rho_max": round(float(final_report.get("rho_max", float(np.max(init_rho)))), 4),
        "final_rho_changed": int(final_report.get("rho_changed", 0)),
        "final_struct": final_struct,
        "final_gate": final_gate,
        "elapsed_sec": int(elapsed),
        "reports": reports,
    }


def format_report_line(mode_key, step, eval_pct, edges, accepts, theta, decay, rho, freq, polarity, init_rho, struct, gate, elapsed, sps, nonfinite):
    theta_min, theta_max = float(theta.min()), float(theta.max())
    decay_min, decay_max = float(decay.min()), float(decay.max())
    rho_changed = int(np.sum(np.abs(rho - init_rho) > 1e-6))
    rho_nonzero = int(np.sum(rho > 0.01))
    freq_mean, freq_std = float(np.mean(freq)), float(np.std(freq))
    inh_count = int(np.sum(polarity < 0))
    exc_count = int(len(polarity) - inh_count)
    main = (
        f"[{mode_key:>15} {step:5d}] eval={eval_pct:5.2f}% edges={edges:4d} "
        f"[Ad={accepts['add']}|Fl={accepts['flip']}|Dc={accepts['decay']}|Rh={accepts['rho']}] "
        f"theta={float(np.mean(theta)):.4f}[{theta_min:.3f},{theta_max:.3f}] "
        f"decay={float(np.mean(decay)):.4f}[{decay_min:.3f},{decay_max:.3f}] "
        f"rho={float(np.mean(rho)):.4f}[{float(np.min(rho)):.3f},{float(np.max(rho)):.3f}] "
        f"nz={rho_nonzero} changed={rho_changed} freq={freq_mean:.4f}±{freq_std:.4f} "
        f"inh={inh_count}/exc={exc_count} {elapsed:.0f}s ({sps:.2f} step/s)"
    )
    if gate["gate_kind"] == "additive_ct":
        detail = (
            f"  gate:add ct_abs={gate['ct_abs_mean']:.4f} ct=[{gate['ct_min']:.3f},{gate['ct_max']:.3f}] "
            f"effT={gate['effective_theta_mean']:.4f}[{gate['effective_theta_min']:.3f},{gate['effective_theta_max']:.3f}] "
            f"struct:recip={struct['reciprocal_pairs']} sink={struct['sink_count']} "
            f"src={struct['source_only_count']} iso={struct['isolated_count']} nonfinite={nonfinite}"
        )
    else:
        detail = (
            f"  gate:graph factor={gate['gate_mean']:.4f}[{gate['gate_min']:.3f},{gate['gate_max']:.3f}] "
            f"effT={gate['effective_theta_mean']:.4f}[{gate['effective_theta_min']:.3f},{gate['effective_theta_max']:.3f}] "
            f"struct:recip={struct['reciprocal_pairs']} sink={struct['sink_count']} "
            f"src={struct['source_only_count']} iso={struct['isolated_count']} nonfinite={nonfinite}"
        )
    return main, detail, {
        "step": int(step),
        "eval_pct": round(float(eval_pct), 2),
        "edges": int(edges),
        "accepts": {key: int(value) for key, value in accepts.items()},
        "theta_mean": round(float(np.mean(theta)), 4),
        "theta_min": round(theta_min, 4),
        "theta_max": round(theta_max, 4),
        "decay_mean": round(float(np.mean(decay)), 4),
        "decay_min": round(decay_min, 4),
        "decay_max": round(decay_max, 4),
        "rho_mean": round(float(np.mean(rho)), 4),
        "rho_min": round(float(np.min(rho)), 4),
        "rho_max": round(float(np.max(rho)), 4),
        "rho_nonzero": int(rho_nonzero),
        "rho_changed": int(rho_changed),
        "freq_mean": round(freq_mean, 4),
        "freq_std": round(freq_std, 4),
        "inh": int(inh_count),
        "exc": int(exc_count),
        "nonfinite_events": int(nonfinite),
        "elapsed_sec": int(elapsed),
        "step_per_sec": round(float(sps), 2),
        "structure": struct,
        "gate": gate,
    }


def train_mode(mode_key, label, schedule, learn_rho, input_projection, output_projection, bp, bigram, all_data, eval_seqs, init_state, args, live_log, json_log, payload):
    H = init_state["H"]
    mask = init_state["mask"].copy()
    theta = init_state["theta"].copy()
    decay = init_state["decay"].copy()
    polarity = init_state["polarity"].copy()
    freq = init_state["freq"].copy()
    phase = init_state["phase"].copy()
    rho = init_state["rho"].copy()
    init_rho = rho.copy()

    accepts = {"add": 0, "flip": 0, "decay": 0, "rho": 0}
    reports = []
    t0 = time.time()

    # Main-process reporting uses the same globals as the worker pool.
    init_worker(
        bp,
        all_data,
        args.seq_len,
        args.train_seqs,
        input_projection,
        output_projection,
        bigram,
        polarity,
        freq,
        phase,
        args.ticks,
        args.input_ticks,
        mode_key,
    )

    append_log(live_log, "")
    append_log(live_log, f"=== MODE {mode_key}: {label} ===")
    print(f"\n=== MODE {mode_key}: {label} ===")
    print(f"schedule={schedule}")
    print(f"rho_mutable={learn_rho}")
    sys.stdout.flush()

    pool = Pool(
        args.workers,
        initializer=init_worker,
        initargs=(
            bp,
            all_data,
            args.seq_len,
            args.train_seqs,
            input_projection,
            output_projection,
            bigram,
            polarity,
            freq,
            phase,
            args.ticks,
            args.input_ticks,
            mode_key,
        ),
    )

    try:
        for step in range(1, args.budget + 1):
            ptype = schedule[(step - 1) % len(schedule)]
            if ptype == "rho" and not learn_rho:
                ptype = "decay"
            if ptype in ("flip", "decay", "rho") and int((mask != 0).sum()) == 0:
                ptype = "add"

            mask_flat = mask.flatten()
            worker_args = [
                (mask_flat, theta.copy(), decay.copy(), rho.copy(), H, 1000 + step * 50 + worker_idx, ptype)
                for worker_idx in range(args.workers)
            ]
            results = pool.map(worker_eval, worker_args)
            best = max(results, key=lambda item: item["delta"])

            if best["delta"] > args.threshold:
                if best["type"] in ("add", "flip") and best["new_mask_flat"] is not None:
                    mask[:] = best["new_mask_flat"].reshape(H, H)
                    accepts[best["type"]] += 1
                elif best["type"] == "decay" and best["new_decay"] is not None:
                    decay[:] = best["new_decay"]
                    accepts["decay"] += 1
                elif best["type"] == "rho" and best["new_rho"] is not None:
                    rho[:] = best["new_rho"]
                    accepts["rho"] += 1

            if step % args.report_every == 0 or step == args.budget:
                elapsed = time.time() - t0
                accs = []
                nonfinite_eval = 0
                for seq in eval_seqs:
                    acc, bad = eval_accuracy(
                        mask, theta, decay, rho, seq, bp, input_projection, output_projection, polarity, freq, phase, mode_key
                    )
                    accs.append(acc)
                    nonfinite_eval += bad
                eval_pct = float(np.mean(accs) * 100.0)
                edges = int((mask != 0).sum())
                struct = structure_stats(mask)
                gate = additive_gate_stats(theta, rho, freq, phase, args.ticks) if mode_key != "graph_exact_rho" else graph_gate_stats(theta, rho, freq, phase, args.ticks)
                line, detail, report_entry = format_report_line(
                    mode_key, step, eval_pct, edges, accepts, theta, decay, rho, freq, polarity, init_rho, struct, gate,
                    elapsed, step / max(elapsed, 1e-8), nonfinite_eval
                )
                reports.append(report_entry)
                append_log(live_log, line)
                append_log(live_log, detail)
                print(line)
                print(detail)
                payload["modes"][mode_key] = {
                    "label": label,
                    "schedule": schedule,
                    "rho_mutable": learn_rho,
                    "reports": reports,
                }
                dump_json(json_log, payload)
                sys.stdout.flush()
    finally:
        pool.terminate()
        pool.join()

    elapsed = time.time() - t0
    accs = []
    nonfinite_final = 0
    for seq in eval_seqs:
        acc, bad = eval_accuracy(mask, theta, decay, rho, seq, bp, input_projection, output_projection, polarity, freq, phase, mode_key)
        accs.append(acc)
        nonfinite_final += bad
    final_eval = float(np.mean(accs))
    final_struct = structure_stats(mask)
    final_gate = additive_gate_stats(theta, rho, freq, phase, args.ticks) if mode_key != "graph_exact_rho" else graph_gate_stats(theta, rho, freq, phase, args.ticks)
    summary = summarize_mode(mode_key, label, reports, init_rho, final_eval, int((mask != 0).sum()), nonfinite_final, final_struct, final_gate, elapsed)
    payload["modes"][mode_key]["summary"] = summary
    dump_json(json_log, payload)
    append_log(live_log, f"FINAL {mode_key}: eval={summary['final_eval_pct']:.2f}% edges={summary['final_edges']} nonfinite={summary['final_nonfinite_events']} elapsed={summary['elapsed_sec']}s")
    print(f"FINAL {mode_key}: eval={summary['final_eval_pct']:.2f}% edges={summary['final_edges']} nonfinite={summary['final_nonfinite_events']} elapsed={summary['elapsed_sec']}s")
    sys.stdout.flush()
    return summary


def build_initial_state(io_dim, hidden_ratio, projection_scale, theta_init, decay_lo, decay_hi):
    ref = SelfWiringGraph(io_dim, hidden_ratio=hidden_ratio, projection_scale=projection_scale, seed=42)
    H = ref.H
    decay_rng = np.random.RandomState(99)
    return {
        "H": H,
        "mask": np.zeros((H, H), dtype=np.float32),
        "theta": np.full(H, theta_init, dtype=np.float32),
        "decay": decay_rng.uniform(decay_lo, decay_hi, H).astype(np.float32),
        "polarity": ref.polarity.astype(np.float32),
        "freq": ref.freq.astype(np.float32).copy(),
        "phase": ref.phase.astype(np.float32).copy(),
        "rho": ref.rho.astype(np.float32).copy(),
        "input_projection": ref.input_projection.astype(np.float32).copy(),
        "output_projection": ref.output_projection.astype(np.float32).copy(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the English C19 truth probe side recipe.")
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
    parser.add_argument("--run-name", default="english_c19_truth_probe")
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

    run_prefix = ROOT / "recipes" / args.run_name
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
            "projection_scale": PROJECTION_SCALE,
            "theta_init": THETA_INIT,
            "decay_init": [DECAY_INIT_LO, DECAY_INIT_HI],
            "modes": list(args.modes),
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

    mode_summaries = []
    for mode_key in args.modes:
        if mode_key == "fixed_additive":
            schedule = BASELINE_SCHEDULE
            learn_rho = False
        else:
            schedule = LEARNABLE_RHO_SCHEDULE
            learn_rho = True
        summary = train_mode(
            mode_key,
            MODE_LABELS[mode_key],
            schedule,
            learn_rho,
            init_state["input_projection"],
            init_state["output_projection"],
            bp,
            bigram,
            all_data,
            eval_seqs,
            init_state,
            args,
            live_log,
            json_log,
            payload,
        )
        mode_summaries.append(summary)

    baseline = next(item for item in mode_summaries if item["mode"] == "fixed_additive")
    best_candidate = baseline
    for item in mode_summaries:
        if item["mode"] == "fixed_additive":
            continue
        if item["final_eval_pct"] > best_candidate["final_eval_pct"]:
            best_candidate = item

    promotion = {
        "baseline_mode": baseline["mode"],
        "baseline_final_eval_pct": baseline["final_eval_pct"],
        "winner_mode": best_candidate["mode"],
        "winner_final_eval_pct": best_candidate["final_eval_pct"],
        "winner_peak_eval_pct": best_candidate["peak_eval_pct"],
        "winner_final_edges": best_candidate["final_edges"],
        "winner_rho_changed": best_candidate["final_rho_changed"],
        "winner_nonfinite_events": best_candidate["final_nonfinite_events"],
        "promote": bool(
            best_candidate["mode"] != "fixed_additive"
            and best_candidate["final_eval_pct"] > baseline["final_eval_pct"]
            and best_candidate["final_nonfinite_events"] == 0
            and best_candidate["final_rho_changed"] > 0
        ),
    }
    promotion["verdict"] = (
        f"PROMOTE {best_candidate['mode']}"
        if promotion["promote"]
        else "KEEP CANONICAL MAINLINE"
    )

    payload["summary_table"] = mode_summaries
    payload["promotion_verdict"] = promotion
    dump_json(json_log, payload)

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for item in mode_summaries:
        struct = item["final_struct"]
        print(
            f"{item['mode']:>15}  final={item['final_eval_pct']:6.2f}%  "
            f"peak={item['peak_eval_pct']:6.2f}%  t_peak={item['time_to_peak_sec']:5d}s  "
            f"edges={item['final_edges']:4d}  rho_changed={item['final_rho_changed']:4d}  "
            f"recip={struct['reciprocal_pairs']:3d} sink={struct['sink_count']:4d} "
            f"src={struct['source_only_count']:4d} iso={struct['isolated_count']:4d}"
        )
    print("-" * 72)
    print(
        f"Promotion verdict: {promotion['verdict']} "
        f"(baseline={promotion['baseline_final_eval_pct']:.2f}% "
        f"winner={promotion['winner_mode']} {promotion['winner_final_eval_pct']:.2f}%)"
    )
    print(f"Live log saved: {live_log}")
    print(f"JSON report saved: {json_log}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
