#!/usr/bin/env python3
"""Manual neuron grower explorer.

Exhaustive (or exhaustive-ish) candidate search for the NEXT neuron to add
to a manually-grown grid3 ensemble. User inspects the ranked table and picks
which candidate to bake in.

Usage:
    python tools/manual_grow_explorer.py --task grid3_center --top-k 25 --max-parents 3
    python tools/manual_grow_explorer.py --task grid3_center --add '{"parents":[4],"weights":[1],"threshold":1}'
    python tools/manual_grow_explorer.py --task grid3_center --show-state
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "target" / "manual_grow"


# ---------------------------------------------------------------------------
# Task definitions (mirror instnct-core/examples/grid3_curriculum.rs label fns)
# ---------------------------------------------------------------------------

def label_grid3_center(b):
    return b[4]

def label_grid3_diagonal(b):
    return 1 if (b[0] and b[4] and b[8]) else 0

def label_grid3_corner(b):
    return 1 if (b[0] or b[2] or b[6] or b[8]) else 0

def label_grid3_horizontal_line(b):
    return 1 if ((b[0] and b[1] and b[2]) or (b[3] and b[4] and b[5]) or (b[6] and b[7] and b[8])) else 0

def label_grid3_vertical_line(b):
    return 1 if ((b[0] and b[3] and b[6]) or (b[1] and b[4] and b[7]) or (b[2] and b[5] and b[8])) else 0

def label_grid3_majority(b):
    return 1 if sum(b) >= 5 else 0

def label_grid3_full_parity(b):
    return sum(b) & 1

def label_grid3_diag_xor(b):
    return (b[0] ^ b[4] ^ b[8])

def label_grid3_symmetry_h(b):
    return 1 if (b[0] == b[2] and b[3] == b[5] and b[6] == b[8]) else 0

def label_grid3_top_heavy(b):
    return 1 if (b[0] + b[1] + b[2]) > (b[6] + b[7] + b[8]) else 0


# grid3 multi-head task progression (each task has 9 bit-heads).
# Level 0: copy (identity) | Level 1: invert | Level 2: shift_right (cyclic)
# Level 3: reflect_h        | Level 4: rotate_90 | Level 5: xor_pair
# Level 6: xor_triple       | Level 7: full_parity (all heads learn 9-bit parity)

def _make_copy_bit_label(i):
    return lambda b, i=i: b[i]

def _make_invert_bit_label(i):
    return lambda b, i=i: 1 - b[i]

def _make_shift_right_bit_label(i):
    return lambda b, i=i: b[(i - 1) % 9]

_REFLECT_H_MAP = [2, 1, 0, 5, 4, 3, 8, 7, 6]
def _make_reflect_h_bit_label(i):
    return lambda b, i=i: b[_REFLECT_H_MAP[i]]

_ROT90_MAP = [6, 3, 0, 7, 4, 1, 8, 5, 2]
def _make_rotate_90_bit_label(i):
    return lambda b, i=i: b[_ROT90_MAP[i]]

def _make_xor_pair_bit_label(i):
    return lambda b, i=i: b[i] ^ b[(i + 1) % 9]

def _make_xor_triple_bit_label(i):
    return lambda b, i=i: b[i] ^ b[(i + 1) % 9] ^ b[(i + 2) % 9]

def _make_full_parity_bit_label(i):
    return lambda b, i=i: sum(b) & 1

TASKS: Dict[str, Tuple[int, Callable[[List[int]], int]]] = {
    "grid3_center": (9, label_grid3_center),
    "grid3_diagonal": (9, label_grid3_diagonal),
    "grid3_corner": (9, label_grid3_corner),
    "grid3_horizontal_line": (9, label_grid3_horizontal_line),
    "grid3_vertical_line": (9, label_grid3_vertical_line),
    "grid3_majority": (9, label_grid3_majority),
    "grid3_full_parity": (9, label_grid3_full_parity),
    "grid3_diag_xor": (9, label_grid3_diag_xor),
    "grid3_symmetry_h": (9, label_grid3_symmetry_h),
    "grid3_top_heavy": (9, label_grid3_top_heavy),
}

# Multi-head progression tasks (9 bit-heads each)
_GRID3_MULTIHEAD_PROGRESSION = [
    ("copy", _make_copy_bit_label),
    ("invert", _make_invert_bit_label),
    ("shift_right", _make_shift_right_bit_label),
    ("reflect_h", _make_reflect_h_bit_label),
    ("rotate_90", _make_rotate_90_bit_label),
    ("xor_pair", _make_xor_pair_bit_label),
    ("xor_triple", _make_xor_triple_bit_label),
    ("full_parity", _make_full_parity_bit_label),
]
for _task_name, _label_maker in _GRID3_MULTIHEAD_PROGRESSION:
    for _i in range(9):
        TASKS[f"grid3_{_task_name}_bit_{_i}"] = (9, _label_maker(_i))


# ---------------------------------------------------------------------------
# Deterministic XorShift32 — mirrors instnct-core Rng
# ---------------------------------------------------------------------------

class Xor32:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF
        if self.s == 0:
            self.s = 1

    def next_u32(self) -> int:
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x & 0xFFFFFFFF
        return self.s

    def next_bit(self) -> int:
        return self.next_u32() & 1

    def next_f32(self) -> float:
        return self.next_u32() / 4294967296.0


# ---------------------------------------------------------------------------
# Data generation — pool of random bit vectors + noisy labels
# ---------------------------------------------------------------------------

def gen_data(task: str, seed: int, n_train: int = 200, n_val: int = 200, n_test: int = 200, noise: float = 0.1):
    """Train set carries noise (label flip p=noise). Val/test are clean — theoretical 100% achievable."""
    n_in, label_fn = TASKS[task]
    rng = Xor32(seed)

    def batch(n, apply_noise: bool):
        inputs = []
        labels = []
        for _ in range(n):
            bits = [rng.next_bit() for _ in range(n_in)]
            lab = label_fn(bits)
            if apply_noise and rng.next_f32() < noise:
                lab = 1 - lab
            inputs.append(bits)
            labels.append(lab)
        return inputs, labels

    tr_x, tr_y = batch(n_train, apply_noise=True)
    va_x, va_y = batch(n_val, apply_noise=False)
    te_x, te_y = batch(n_test, apply_noise=False)
    return (tr_x, tr_y), (va_x, va_y), (te_x, te_y), n_in


# ---------------------------------------------------------------------------
# Neuron evaluation
# ---------------------------------------------------------------------------

def neuron_eval(parents, weights, threshold, full_sigs):
    """Evaluate one neuron. full_sigs is the concatenation of input bits + hidden bits."""
    dot = 0
    for p, w in zip(parents, weights):
        if w != 0:
            dot += w * full_sigs[p]
    return 1 if dot >= threshold else 0


def build_sig_vector(input_bits, hidden_outputs, n_in):
    """Return [input_bits..., hidden0, hidden1, ...] so that parent indices resolve globally."""
    return input_bits + hidden_outputs


def hidden_outputs_for_sample(baked_neurons, input_bits, n_in):
    """Compute all hidden neuron outputs for one input sample, in order."""
    outs = []
    for n in baked_neurons:
        sigs = input_bits + outs  # previously-baked hidden outputs seen as sig suffix
        outs.append(neuron_eval(n["parents"], n["weights"], n["threshold"], sigs))
    return outs


def ensemble_predict(baked_neurons, input_bits, n_in) -> int:
    """AdaBoost weighted vote over baked neurons. Returns 0/1 prediction."""
    if not baked_neurons:
        return 0  # degenerate, no info
    outs = hidden_outputs_for_sample(baked_neurons, input_bits, n_in)
    score = 0.0
    for n, o in zip(baked_neurons, outs):
        score += n["alpha"] * (1.0 if o == 1 else -1.0)
    return 1 if score >= 0 else 0


def ensemble_accuracy(baked_neurons, inputs, labels, n_in) -> float:
    if not inputs:
        return 0.0
    ok = 0
    for x, y in zip(inputs, labels):
        if ensemble_predict(baked_neurons, x, n_in) == y:
            ok += 1
    return 100.0 * ok / len(inputs)


# ---------------------------------------------------------------------------
# Scoring a candidate neuron against an existing baked ensemble
# ---------------------------------------------------------------------------

def _compute_sample_weights(baked_neurons, inputs, labels, n_in):
    """AdaBoost-style sample reweighting after training on baked set."""
    n = len(inputs)
    if n == 0:
        return []
    w = [1.0 / n] * n
    if not baked_neurons:
        return w
    # Reweight after each baked neuron was added, in order.
    for idx, neuron in enumerate(baked_neurons):
        partial = baked_neurons[: idx + 1]
        errs = []
        for x, y in zip(inputs, labels):
            outs = hidden_outputs_for_sample(partial, x, n_in)
            vote = outs[-1]  # this specific neuron's vote
            errs.append(0 if vote == y else 1)
        err_rate = sum(wi * e for wi, e in zip(w, errs))
        if err_rate <= 0 or err_rate >= 1:
            continue
        alpha = 0.5 * math.log((1 - err_rate) / err_rate)
        new_w = [
            wi * math.exp(alpha if e else -alpha)
            for wi, e in zip(w, errs)
        ]
        s = sum(new_w)
        if s > 0:
            w = [wi / s for wi in new_w]
    return w


def score_candidate(parents, weights, threshold, tr_inputs, tr_labels, va_inputs, va_labels, baked_neurons, n_in, sample_weights):
    """Return dict with train_acc, val_acc, delta (ensemble val delta), alpha, err."""
    hidden_tr = [hidden_outputs_for_sample(baked_neurons, x, n_in) for x in tr_inputs]
    hidden_va = [hidden_outputs_for_sample(baked_neurons, x, n_in) for x in va_inputs]

    tr_sig = [x + h for x, h in zip(tr_inputs, hidden_tr)]
    va_sig = [x + h for x, h in zip(va_inputs, hidden_va)]

    tr_preds = [neuron_eval(parents, weights, threshold, s) for s in tr_sig]
    va_preds = [neuron_eval(parents, weights, threshold, s) for s in va_sig]

    tr_acc = 100.0 * sum(1 for p, y in zip(tr_preds, tr_labels) if p == y) / len(tr_labels)
    va_acc = 100.0 * sum(1 for p, y in zip(va_preds, va_labels) if p == y) / len(va_labels)

    err = sum(w * (0 if p == y else 1) for w, p, y in zip(sample_weights, tr_preds, tr_labels))
    if err <= 0:
        alpha = 5.0
    elif err >= 1:
        alpha = -5.0
    else:
        alpha = 0.5 * math.log((1 - err) / err)

    baseline_va_acc = ensemble_accuracy(baked_neurons, va_inputs, va_labels, n_in)
    trial_neuron = {"parents": parents, "weights": weights, "threshold": threshold, "alpha": alpha}
    trial_va_acc = ensemble_accuracy(baked_neurons + [trial_neuron], va_inputs, va_labels, n_in)
    delta = trial_va_acc - baseline_va_acc

    return {
        "train_acc": tr_acc,
        "val_acc": va_acc,
        "alpha": alpha,
        "err": err,
        "ensemble_val_before": baseline_va_acc,
        "ensemble_val_after": trial_va_acc,
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

def iter_1parent(n_sig: int):
    for p in range(n_sig):
        for w in (-1, 1):
            # threshold choices that give non-trivial outputs for binary parent
            for t in (0, 1):
                if w == -1 and t == 1:
                    continue  # output is always 0 (dot = -b[p] ∈ {-1,0}, never >=1)
                if w == 1 and t == 0:
                    continue  # output always 1 (dot ∈ {0,1}, both >=0)
                yield tuple([p]), (w,), t


def iter_kparent(n_sig: int, k: int, threshold_range):
    """Yield (parents, weights, threshold) for exactly k parents with non-zero weights."""
    parent_combos = itertools.combinations(range(n_sig), k)
    for parents in parent_combos:
        # Each weight in {-1,+1} (skip 0 here — zero weights would be equivalent to fewer parents)
        for weights in itertools.product((-1, 1), repeat=k):
            for t in threshold_range:
                yield tuple(parents), tuple(weights), t


# ---------------------------------------------------------------------------
# State file I/O
# ---------------------------------------------------------------------------

def state_path_for(task: str) -> Path:
    return STATE_DIR / task / "state.json"


def load_state(task: str):
    p = state_path_for(task)
    if not p.exists():
        return {"task": task, "neurons": []}
    return json.loads(p.read_text())


def save_state(task: str, state: dict):
    p = state_path_for(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fmt_parents(parents, n_in):
    out = []
    for p in parents:
        if p < n_in:
            out.append(f"x{p}")
        else:
            out.append(f"N{p - n_in}")
    return "[" + ",".join(out) + "]"


def fmt_weights(weights):
    sym = {-1: "-", 0: "0", 1: "+"}
    return "".join(sym[w] for w in weights)


def render_table(rows, n_in, top_k):
    print()
    print(f"| #  | parents                          | w         | thr | train% | val%  | dV_ens  | alpha  |")
    print(f"|---:|:---------------------------------|:----------|----:|-------:|------:|--------:|-------:|")
    for i, r in enumerate(rows[:top_k], 1):
        p = fmt_parents(r["parents"], n_in)
        w = fmt_weights(r["weights"])
        print(
            f"| {i:>2} | {p:<32} | {w:<9} | {r['threshold']:>3} | "
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
    ap.add_argument("--add", type=str, help='JSON neuron spec: {"parents":[...],"weights":[...],"threshold":N}')
    ap.add_argument("--show-state", action="store_true")
    ap.add_argument("--reset", action="store_true", help="Wipe the baked state for this task")
    args = ap.parse_args()

    if args.reset:
        p = state_path_for(args.task)
        if p.exists():
            p.unlink()
        print(f"[reset] wiped {p}")
        return 0

    state = load_state(args.task)
    baked = state["neurons"]

    if args.show_state:
        print(f"\n=== baked state for {args.task} ===")
        if not baked:
            print("(empty — 0 neurons)")
        else:
            for i, n in enumerate(baked):
                print(f"  N{i}: parents={n['parents']} weights={n['weights']} thr={n['threshold']} alpha={n['alpha']:+.4f}")
        (train, val, test, n_in) = gen_data(args.task, args.seed)
        ens_train = ensemble_accuracy(baked, train[0], train[1], n_in)
        ens_val = ensemble_accuracy(baked, val[0], val[1], n_in)
        ens_test = ensemble_accuracy(baked, test[0], test[1], n_in)
        print(f"ensemble train={ens_train:.2f}%  val={ens_val:.2f}%  test={ens_test:.2f}%")
        return 0

    if args.add:
        spec = json.loads(args.add)
        parents = list(spec["parents"])
        weights = list(spec["weights"])
        thr = int(spec["threshold"])
        (train, val, _, n_in) = gen_data(args.task, args.seed)
        sw = _compute_sample_weights(baked, train[0], train[1], n_in)
        # Compute alpha for the new neuron
        hidden_tr = [hidden_outputs_for_sample(baked, x, n_in) for x in train[0]]
        tr_sig = [x + h for x, h in zip(train[0], hidden_tr)]
        preds = [neuron_eval(parents, weights, thr, s) for s in tr_sig]
        err = sum(w * (0 if p == y else 1) for w, p, y in zip(sw, preds, train[1]))
        if err <= 0:
            alpha = 5.0
        elif err >= 1:
            alpha = -5.0
        else:
            alpha = 0.5 * math.log((1 - err) / err)
        new_neuron = {"parents": parents, "weights": weights, "threshold": thr, "alpha": alpha}
        baked.append(new_neuron)
        state["neurons"] = baked
        save_state(args.task, state)
        n_idx = len(baked) - 1
        ens_val = ensemble_accuracy(baked, val[0], val[1], n_in)
        print(f"\n[add] N{n_idx} baked: parents={parents} weights={weights} thr={thr} alpha={alpha:+.4f}")
        print(f"      ensemble val now: {ens_val:.2f}%")
        print(f"      state file: {state_path_for(args.task)}")
        return 0

    # Default: exhaustive search next candidate
    (train, val, _, n_in) = gen_data(args.task, args.seed)
    n_sig = n_in + len(baked)
    sw = _compute_sample_weights(baked, train[0], train[1], n_in)

    baseline_val = ensemble_accuracy(baked, val[0], val[1], n_in)
    print(f"\n=== manual_grow_explorer — task={args.task} seed={args.seed} ===")
    print(f"baked neurons: {len(baked)}  n_in: {n_in}  n_sig_available: {n_sig}")
    print(f"baseline ensemble val_acc: {baseline_val:.2f}%")

    candidates = []

    # 1-parent exhaustive
    for parents, weights, thr in iter_1parent(n_sig):
        res = score_candidate(parents, weights, thr, train[0], train[1], val[0], val[1], baked, n_in, sw)
        res["parents"] = parents
        res["weights"] = weights
        res["threshold"] = thr
        candidates.append(res)

    # k-parent for k in 2..max_parents
    for k in range(2, args.max_parents + 1):
        # Threshold range: at most ±k
        thr_range = list(range(-k + 1, k + 1))
        for parents, weights, thr in iter_kparent(n_sig, k, thr_range):
            res = score_candidate(parents, weights, thr, train[0], train[1], val[0], val[1], baked, n_in, sw)
            res["parents"] = parents
            res["weights"] = weights
            res["threshold"] = thr
            candidates.append(res)

    # Sort: highest delta first, tiebreak on val_acc then fewer parents then lower threshold
    candidates.sort(key=lambda r: (-r["delta"], -r["val_acc"], len(r["parents"]), r["threshold"]))

    total = len(candidates)
    print(f"evaluated {total} candidates (1-parent exhaustive + up to {args.max_parents}-parent)")

    render_table(candidates, n_in, args.top_k)

    print(f"To bake a choice, re-run with: --add '<json spec>'")
    print(f"Example: --add '{{\"parents\":[4],\"weights\":[1],\"threshold\":1}}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
