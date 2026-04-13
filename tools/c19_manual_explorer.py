#!/usr/bin/env python3
"""Manual c19 neuron grower explorer.

Mirrors tools/manual_grow_explorer.py but uses the C19 activation function
with a sparse 10x6=60 quant grid search per candidate for (c, rho).

Usage:
    python tools/c19_manual_explorer.py --task grid3_center --top-k 25 --max-parents 3
    python tools/c19_manual_explorer.py --task grid3_center --add '{"parents":[1,4],"weights":[0,1],"threshold":1}'
    python tools/c19_manual_explorer.py --task grid3_center --show-state
    python tools/c19_manual_explorer.py --task grid3_center --reset
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

# Reuse data-gen scaffolding + tasks from baseline
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manual_grow_explorer import (
    TASKS, Xor32, gen_data, iter_1parent,
    fmt_parents, fmt_weights,
)

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "target" / "c19_manual_grow"


# ---------------------------------------------------------------------------
# C19 activation — verbatim Python port of instnct-core c19_grower.rs c19()
# ---------------------------------------------------------------------------

def c19(x: float, c: float, rho: float) -> float:
    c = max(c, 0.1)
    rho = max(rho, 0.0)
    l = 6.0 * c
    if x >= l:
        return x - l
    if x <= -l:
        return x + l
    scaled = x / c
    n = math.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    sgn = 1.0 if (int(n) % 2 == 0) else -1.0
    return c * (sgn * h + rho * h * h)


# ---------------------------------------------------------------------------
# Quant grid (matches C_GRID, RHO_GRID in c19_grower.rs)
# ---------------------------------------------------------------------------

C_GRID = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
RHO_GRID = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]


def c19_weighted_mse(dots: List[int], targets: List[float], sw: List[float], c: float, rho: float) -> float:
    loss = 0.0
    for d, t, w in zip(dots, targets, sw):
        out = c19(float(d), c, rho)
        e = out - t
        loss += w * e * e
    return loss


def quant_grid_search(dots: List[int], targets: List[float], sw: List[float]) -> Tuple[float, float, float]:
    """Return (best_c, best_rho, best_loss) over the 10x6=60 grid."""
    best = (1.0, 0.0, float("inf"))
    for c_val in C_GRID:
        for r_val in RHO_GRID:
            l = c19_weighted_mse(dots, targets, sw, c_val, r_val)
            if l < best[2]:
                best = (c_val, r_val, l)
    return best


def bake_lut_py(weights: List[int], c: float, rho: float) -> Tuple[List[float], int]:
    abs_sum = sum(abs(w) for w in weights)
    min_dot = -abs_sum
    lut = [c19(float(d), c, rho) for d in range(min_dot, abs_sum + 1)]
    return lut, min_dot


# ---------------------------------------------------------------------------
# C19 neuron evaluation (LUT-based, binary thresholded for downstream signals)
# ---------------------------------------------------------------------------

def lut_lookup(lut: List[float], lut_min_dot: int, dot: int) -> float:
    idx = dot - lut_min_dot
    if idx < 0 or idx >= len(lut):
        return 0.0
    return lut[idx]


def neuron_eval_c19_pred(neuron: dict, full_sigs: List[int]) -> int:
    dot = 0
    for w, p in zip(neuron["weights"], neuron["parents"]):
        if w != 0:
            dot += w * full_sigs[p]
    v = lut_lookup(neuron["lut"], neuron["lut_min_dot"], dot)
    return 1 if v >= 0.0 else 0


def hidden_outputs_for_sample_c19(baked_neurons, input_bits, n_in):
    outs = []
    for n in baked_neurons:
        sigs = input_bits + outs
        outs.append(neuron_eval_c19_pred(n, sigs))
    return outs


def ensemble_predict_c19(baked_neurons, input_bits, n_in) -> int:
    # Rust c19_grower soft voting: score = sum(alpha_i * eval_lut(dot_i)),
    # where eval_lut returns the continuous c19 LUT value (not thresholded).
    # Hidden signal chain stays binary (eval_lut_pred) so downstream dots stay integer.
    if not baked_neurons:
        return 0
    score = 0.0
    sigs = list(input_bits)
    for neuron in baked_neurons:
        dot = 0
        for w, p in zip(neuron["weights"], neuron["parents"]):
            if w != 0:
                dot += w * sigs[p]
        lut_val = lut_lookup(neuron["lut"], neuron["lut_min_dot"], dot)
        score += neuron["alpha"] * lut_val
        sigs.append(1 if lut_val >= 0.0 else 0)
    return 1 if score >= 0 else 0


def ensemble_accuracy_c19(baked_neurons, inputs, labels, n_in) -> float:
    if not inputs:
        return 0.0
    ok = 0
    for x, y in zip(inputs, labels):
        if ensemble_predict_c19(baked_neurons, x, n_in) == y:
            ok += 1
    return 100.0 * ok / len(inputs)


def compute_sample_weights_c19(baked_neurons, inputs, labels, n_in):
    n = len(inputs)
    if n == 0:
        return []
    w = [1.0 / n] * n
    if not baked_neurons:
        return w
    for idx, neuron in enumerate(baked_neurons):
        partial = baked_neurons[: idx + 1]
        errs = []
        for x, y in zip(inputs, labels):
            outs = hidden_outputs_for_sample_c19(partial, x, n_in)
            vote = outs[-1]
            errs.append(0 if vote == y else 1)
        err_rate = sum(wi * e for wi, e in zip(w, errs))
        if err_rate <= 0 or err_rate >= 1:
            continue
        alpha = 0.5 * math.log((1 - err_rate) / err_rate)
        new_w = [wi * math.exp(alpha if e else -alpha) for wi, e in zip(w, errs)]
        s = sum(new_w)
        if s > 0:
            w = [wi / s for wi in new_w]
    return w


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def score_candidate_c19(parents, weights, threshold, tr_inputs, tr_labels, va_inputs, va_labels,
                        baked_neurons, n_in, sample_weights,
                        tr_sig=None, va_sig=None, baseline_va_acc=None):
    """Return dict with val_acc, train_acc, c, rho, lut, alpha, delta.

    tr_sig / va_sig precomputed input+hidden signal vectors, baseline_va_acc precomputed
    — pass these from the caller to avoid recomputing per candidate.
    """
    if tr_sig is None:
        hidden_tr = [hidden_outputs_for_sample_c19(baked_neurons, x, n_in) for x in tr_inputs]
        tr_sig = [x + h for x, h in zip(tr_inputs, hidden_tr)]
    if va_sig is None:
        hidden_va = [hidden_outputs_for_sample_c19(baked_neurons, x, n_in) for x in va_inputs]
        va_sig = [x + h for x, h in zip(va_inputs, hidden_va)]
    if baseline_va_acc is None:
        baseline_va_acc = ensemble_accuracy_c19(baked_neurons, va_inputs, va_labels, n_in)

    # Integer dots (constant across (c, rho) grid)
    tr_dots = []
    for s in tr_sig:
        d = 0
        for w, p in zip(weights, parents):
            if w != 0:
                d += w * s[p]
        tr_dots.append(d)
    va_dots = []
    for s in va_sig:
        d = 0
        for w, p in zip(weights, parents):
            if w != 0:
                d += w * s[p]
        va_dots.append(d)

    # Targets for MSE loss: +1/-1 per noised label
    tr_targets = [1.0 if y == 1 else -1.0 for y in tr_labels]

    # Sparse quant grid search
    best_c, best_rho, best_loss = quant_grid_search(tr_dots, tr_targets, sample_weights)

    # Bake LUT
    lut, lut_min_dot = bake_lut_py(list(weights), best_c, best_rho)

    # Predictions with baked LUT (thresholded at 0)
    tr_preds = [1 if lut_lookup(lut, lut_min_dot, d) >= 0.0 else 0 for d in tr_dots]
    va_preds = [1 if lut_lookup(lut, lut_min_dot, d) >= 0.0 else 0 for d in va_dots]

    tr_acc = 100.0 * sum(1 for p, y in zip(tr_preds, tr_labels) if p == y) / len(tr_labels)
    va_acc = 100.0 * sum(1 for p, y in zip(va_preds, va_labels) if p == y) / len(va_labels)

    err = sum(w * (0 if p == y else 1) for w, p, y in zip(sample_weights, tr_preds, tr_labels))
    if err <= 0:
        alpha = 5.0
    elif err >= 1:
        alpha = -5.0
    else:
        alpha = 0.5 * math.log((1 - err) / err)

    trial_neuron = {
        "parents": list(parents),
        "weights": list(weights),
        "threshold": int(threshold),
        "c": best_c,
        "rho": best_rho,
        "lut": lut,
        "lut_min_dot": lut_min_dot,
        "alpha": alpha,
    }
    trial_va_acc = ensemble_accuracy_c19(baked_neurons + [trial_neuron], va_inputs, va_labels, n_in)
    delta = trial_va_acc - baseline_va_acc

    return {
        "train_acc": tr_acc,
        "val_acc": va_acc,
        "alpha": alpha,
        "err": err,
        "c": best_c,
        "rho": best_rho,
        "best_loss": best_loss,
        "lut": lut,
        "lut_min_dot": lut_min_dot,
        "ensemble_val_before": baseline_va_acc,
        "ensemble_val_after": trial_va_acc,
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# Candidate enumeration — threshold is unused by c19 eval, so we fix it to 0
# and skip threshold iteration (unlike the baseline tool).
# ---------------------------------------------------------------------------

def iter_kparent_c19(n_sig: int, k: int):
    for parents in itertools.combinations(range(n_sig), k):
        for weights in itertools.product((-1, 1), repeat=k):
            yield tuple(parents), tuple(weights), 0


def iter_1parent_c19(n_sig: int):
    # For c19, a single-parent neuron with dot in {0, +1} or {-1, 0} has
    # c19(0) = 0 identically. The only "useful" 1-parent is w=+1 because
    # the w=-1 case maps to c19(-1)/c19(0) which is the same magnitude.
    # We still enumerate both w=+1 and w=-1 for completeness, threshold fixed.
    for p in range(n_sig):
        for w in (-1, 1):
            yield (p,), (w,), 0


# ---------------------------------------------------------------------------
# State file I/O (separate from baseline)
# ---------------------------------------------------------------------------

def state_path_for(task: str) -> Path:
    return STATE_DIR / task / "state.json"


def load_state(task: str):
    p = state_path_for(task)
    if not p.exists():
        return {"task": task, "activation": "c19", "neurons": []}
    return json.loads(p.read_text())


def save_state(task: str, state: dict):
    p = state_path_for(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_table(rows, n_in, top_k):
    print()
    print(f"| #  | parents                          | w         | c    | rho  | train% | val%  | dV_ens  | alpha  |")
    print(f"|---:|:---------------------------------|:----------|-----:|-----:|-------:|------:|--------:|-------:|")
    for i, r in enumerate(rows[:top_k], 1):
        p = fmt_parents(r["parents"], n_in)
        w = fmt_weights(r["weights"])
        print(
            f"| {i:>2} | {p:<32} | {w:<9} | {r['c']:>4.2f} | {r['rho']:>4.2f} | "
            f"{r['train_acc']:>6.2f} | {r['val_acc']:>5.2f} | {r['delta']:>+8.2f} | {r['alpha']:>+6.3f} |"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--max-parents", type=int, default=3)
    ap.add_argument("--add", type=str,
                    help='JSON neuron spec: {"parents":[...],"weights":[...],"threshold":N}  (threshold ignored by c19 eval)')
    ap.add_argument("--show-state", action="store_true")
    ap.add_argument("--reset", action="store_true", help="Wipe the baked c19 state for this task")
    args = ap.parse_args()

    if args.reset:
        p = state_path_for(args.task)
        if p.exists():
            p.unlink()
        print(f"[reset] wiped {p}")
        return 0

    state = load_state(args.task)
    baked = state.get("neurons", [])

    if args.show_state:
        print(f"\n=== baked c19 state for {args.task} ===")
        if not baked:
            print("(empty - 0 neurons)")
        else:
            for i, n in enumerate(baked):
                lut_preview = ",".join(f"{v:+.3f}" for v in n["lut"])
                print(f"  N{i}: parents={n['parents']} weights={n['weights']} "
                      f"c={n['c']:.4f} rho={n['rho']:.4f} alpha={n['alpha']:+.4f}")
                print(f"       lut[dot={n['lut_min_dot']}..]=[{lut_preview}]")
        (train, val, test, n_in) = gen_data(args.task, args.seed)
        ens_train = ensemble_accuracy_c19(baked, train[0], train[1], n_in)
        ens_val = ensemble_accuracy_c19(baked, val[0], val[1], n_in)
        ens_test = ensemble_accuracy_c19(baked, test[0], test[1], n_in)
        print(f"ensemble train={ens_train:.2f}%  val={ens_val:.2f}%  test={ens_test:.2f}%")
        return 0

    if args.add:
        spec = json.loads(args.add)
        parents = list(spec["parents"])
        weights = list(spec["weights"])
        thr = int(spec.get("threshold", 0))
        (train, val, _, n_in) = gen_data(args.task, args.seed)
        sw = compute_sample_weights_c19(baked, train[0], train[1], n_in)
        baseline_va_acc = ensemble_accuracy_c19(baked, val[0], val[1], n_in)
        res = score_candidate_c19(
            tuple(parents), tuple(weights), thr,
            train[0], train[1], val[0], val[1],
            baked, n_in, sw, baseline_va_acc=baseline_va_acc,
        )
        new_neuron = {
            "parents": parents,
            "weights": weights,
            "threshold": thr,
            "c": res["c"],
            "rho": res["rho"],
            "lut": res["lut"],
            "lut_min_dot": res["lut_min_dot"],
            "alpha": res["alpha"],
        }
        baked.append(new_neuron)
        state["neurons"] = baked
        state["activation"] = "c19"
        save_state(args.task, state)
        n_idx = len(baked) - 1
        ens_val = ensemble_accuracy_c19(baked, val[0], val[1], n_in)
        print(f"\n[add] N{n_idx} baked: parents={parents} weights={weights} thr={thr}")
        print(f"      c={res['c']:.4f}  rho={res['rho']:.4f}  alpha={res['alpha']:+.4f}")
        print(f"      LUT[dot={res['lut_min_dot']}..]={[f'{v:+.3f}' for v in res['lut']]}")
        print(f"      ensemble val now: {ens_val:.2f}%")
        print(f"      state file: {state_path_for(args.task)}")
        return 0

    # Default: exhaustive search
    (train, val, _, n_in) = gen_data(args.task, args.seed)
    n_sig = n_in + len(baked)
    sw = compute_sample_weights_c19(baked, train[0], train[1], n_in)
    baseline_va_acc = ensemble_accuracy_c19(baked, val[0], val[1], n_in)

    print(f"\n=== c19_manual_explorer - task={args.task} seed={args.seed} ===")
    print(f"baked neurons: {len(baked)}  n_in: {n_in}  n_sig_available: {n_sig}")
    print(f"baseline ensemble val_acc: {baseline_va_acc:.2f}%")

    # Precompute hidden signals once (candidate-invariant)
    hidden_tr = [hidden_outputs_for_sample_c19(baked, x, n_in) for x in train[0]]
    hidden_va = [hidden_outputs_for_sample_c19(baked, x, n_in) for x in val[0]]
    tr_sig = [x + h for x, h in zip(train[0], hidden_tr)]
    va_sig = [x + h for x, h in zip(val[0], hidden_va)]

    candidates = []

    for parents, weights, thr in iter_1parent_c19(n_sig):
        res = score_candidate_c19(
            parents, weights, thr,
            train[0], train[1], val[0], val[1],
            baked, n_in, sw,
            tr_sig=tr_sig, va_sig=va_sig, baseline_va_acc=baseline_va_acc,
        )
        res["parents"] = parents
        res["weights"] = weights
        res["threshold"] = thr
        candidates.append(res)

    for k in range(2, args.max_parents + 1):
        for parents, weights, thr in iter_kparent_c19(n_sig, k):
            res = score_candidate_c19(
                parents, weights, thr,
                train[0], train[1], val[0], val[1],
                baked, n_in, sw,
                tr_sig=tr_sig, va_sig=va_sig, baseline_va_acc=baseline_va_acc,
            )
            res["parents"] = parents
            res["weights"] = weights
            res["threshold"] = thr
            candidates.append(res)

    # Sort: highest delta first, tiebreak on val_acc then fewer parents
    candidates.sort(key=lambda r: (-r["delta"], -r["val_acc"], len(r["parents"])))

    total = len(candidates)
    print(f"evaluated {total} candidates (1-parent exhaustive + up to {args.max_parents}-parent)")
    print(f"quant grid: {len(C_GRID)}x{len(RHO_GRID)} = {len(C_GRID)*len(RHO_GRID)} (c,rho) pairs per candidate")

    render_table(candidates, n_in, args.top_k)

    print(f"To bake a choice, re-run with: --add '<json spec>'")
    print(f"Example: --add '{{\"parents\":[1,4],\"weights\":[0,1],\"threshold\":1}}'")
    print(f"(threshold is legacy - c19 eval ignores it)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
