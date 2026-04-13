#!/usr/bin/env python3
"""Overnight grower orchestrator: one-step worker for the /loop skill.

Each invocation does exactly ONE of: (a) bake the next neuron on the current
head via c19_manual_explorer exhaustive search + intelligent top-1 pick,
(b) advance to the next bit head if the current one hit 100%, or
(c) advance to the next task if all 9 heads of the current task are done.

State: target/overnight_state.json (task_idx, bit_idx, counters).
Log:   .claude/research/swarm_logs/overnight_build.log
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from manual_grow_explorer import gen_data
from c19_manual_explorer import (
    score_candidate_c19,
    iter_1parent_c19,
    iter_kparent_c19,
    compute_sample_weights_c19,
    ensemble_accuracy_c19,
    hidden_outputs_for_sample_c19,
    fmt_parents,
    fmt_weights,
    state_path_for,
)

STATE_FILE = ROOT / "target" / "overnight_state.json"
LOG_FILE = ROOT / ".claude" / "research" / "swarm_logs" / "overnight_build.log"

TASK_PROGRESSION = [
    "grid3_copy",
    "grid3_invert",
    "grid3_shift_right",
    "grid3_reflect_h",
    "grid3_rotate_90",
    "grid3_xor_pair",
    "grid3_xor_triple",
    "grid3_full_parity",
]

MAX_NEURONS_PER_HEAD = 5
MAX_PARENTS = 3
SEED = 42
MIN_DELTA_TO_BAKE = 0.25


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_overnight_state():
    if not STATE_FILE.exists():
        return {
            "task_idx": 0,
            "bit_idx": 0,
            "total_bakes": 0,
            "total_advances": 0,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "history": [],
        }
    return json.loads(STATE_FILE.read_text())


def save_overnight_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def current_head_name(state):
    task = TASK_PROGRESSION[state["task_idx"]]
    return f"{task}_bit_{state['bit_idx']}"


def load_head_neurons(head_name: str):
    p = state_path_for(head_name)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("neurons", [])


def bake_candidate(head_name: str, pick: dict):
    p = state_path_for(head_name)
    if p.exists():
        state = json.loads(p.read_text())
    else:
        state = {"task": head_name, "activation": "c19", "neurons": []}
    new_neuron = {
        "parents": list(pick["parents"]),
        "weights": list(pick["weights"]),
        "threshold": int(pick["threshold"]),
        "c": pick["c"],
        "rho": pick["rho"],
        "lut": pick["lut"],
        "lut_min_dot": pick["lut_min_dot"],
        "alpha": pick["alpha"],
    }
    state.setdefault("neurons", []).append(new_neuron)
    state["activation"] = "c19"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def intelligent_pick(candidates: list):
    """From a pre-sorted candidate list (top-5 by delta), apply tiebreaks:
    1. Highest delta (primary)
    2. Fewest parents
    3. Simpler (c, rho) values (preferring integer-like c in {1.0, 1.5, 2.0} and rho in {0, 1, 2, 4})
    4. Higher positive alpha
    """
    if not candidates:
        return None

    def simpler_score(r):
        c_round = round(r["c"], 1)
        rho_round = round(r["rho"], 1)
        c_pref = 0 if c_round in (1.0, 1.5, 2.0, 3.0) else 1
        rho_pref = 0 if rho_round in (0.0, 1.0, 2.0, 4.0) else 1
        return c_pref + rho_pref

    return sorted(
        candidates,
        key=lambda r: (-r["delta"], len(r["parents"]), simpler_score(r), -abs(r["alpha"]))
    )[0]


def run_exhaustive(head_name: str, baked: list, n_in: int):
    (train, val, test, _) = gen_data(head_name, SEED)
    n_sig = n_in + len(baked)
    sw = compute_sample_weights_c19(baked, train[0], train[1], n_in)
    baseline_va = ensemble_accuracy_c19(baked, val[0], val[1], n_in)
    hidden_tr = [hidden_outputs_for_sample_c19(baked, x, n_in) for x in train[0]]
    hidden_va = [hidden_outputs_for_sample_c19(baked, x, n_in) for x in val[0]]
    tr_sig = [x + h for x, h in zip(train[0], hidden_tr)]
    va_sig = [x + h for x, h in zip(val[0], hidden_va)]

    cands = []
    for p, w, t in iter_1parent_c19(n_sig):
        r = score_candidate_c19(p, w, t, train[0], train[1], val[0], val[1], baked, n_in, sw,
                                tr_sig=tr_sig, va_sig=va_sig, baseline_va_acc=baseline_va)
        r.update({"parents": p, "weights": w, "threshold": t})
        cands.append(r)
    for k in range(2, MAX_PARENTS + 1):
        for p, w, t in iter_kparent_c19(n_sig, k):
            r = score_candidate_c19(p, w, t, train[0], train[1], val[0], val[1], baked, n_in, sw,
                                    tr_sig=tr_sig, va_sig=va_sig, baseline_va_acc=baseline_va)
            r.update({"parents": p, "weights": w, "threshold": t})
            cands.append(r)
    cands.sort(key=lambda r: (-r["delta"], -r["val_acc"], len(r["parents"])))
    return cands, baseline_va, (train, val, test, n_in)


def step():
    state = load_overnight_state()
    if state["task_idx"] >= len(TASK_PROGRESSION):
        log("ALL_TASKS_COMPLETE")
        return "done"

    head_name = current_head_name(state)
    task = TASK_PROGRESSION[state["task_idx"]]
    bit = state["bit_idx"]

    baked = load_head_neurons(head_name)
    (_, val, test, n_in) = gen_data(head_name, SEED)
    cur_val = ensemble_accuracy_c19(baked, val[0], val[1], n_in)
    cur_test = ensemble_accuracy_c19(baked, test[0], test[1], n_in)

    # 1. Head already 100% -> advance
    if cur_val >= 100.0 - 1e-6:
        log(f"HEAD_DONE {head_name}: N={len(baked)} val={cur_val:.1f}% test={cur_test:.1f}% -> advance bit")
        state["bit_idx"] += 1
        state["total_advances"] += 1
        if state["bit_idx"] >= 9:
            state["task_idx"] += 1
            state["bit_idx"] = 0
            if state["task_idx"] >= len(TASK_PROGRESSION):
                log("ALL_TASKS_COMPLETE")
            else:
                log(f"TASK_ADVANCE {task} -> {TASK_PROGRESSION[state['task_idx']]}")
        save_overnight_state(state)
        return "head-done-advance"

    # 2. Hit max neurons -> skip head
    if len(baked) >= MAX_NEURONS_PER_HEAD:
        log(f"HEAD_STUCK {head_name}: N={len(baked)} val={cur_val:.1f}% -> skip bit")
        state["bit_idx"] += 1
        state["total_advances"] += 1
        if state["bit_idx"] >= 9:
            state["task_idx"] += 1
            state["bit_idx"] = 0
            if state["task_idx"] >= len(TASK_PROGRESSION):
                log("ALL_TASKS_COMPLETE")
            else:
                log(f"TASK_ADVANCE {task} -> {TASK_PROGRESSION[state['task_idx']]}")
        save_overnight_state(state)
        return "head-stuck-advance"

    # 3. Run exhaustive search
    t_start = time.time()
    cands, baseline_va, data = run_exhaustive(head_name, baked, n_in)
    t_search = time.time() - t_start

    if not cands:
        log(f"NO_CANDIDATES {head_name} -> skip")
        state["bit_idx"] += 1
        save_overnight_state(state)
        return "no-candidates"

    top_5 = cands[:5]
    pick = intelligent_pick(top_5)

    if pick["delta"] < MIN_DELTA_TO_BAKE:
        log(f"PLATEAU {head_name}: top delta={pick['delta']:+.2f}pp (below {MIN_DELTA_TO_BAKE}) -> skip bit  ({t_search:.1f}s search)")
        state["bit_idx"] += 1
        state["total_advances"] += 1
        if state["bit_idx"] >= 9:
            state["task_idx"] += 1
            state["bit_idx"] = 0
            if state["task_idx"] >= len(TASK_PROGRESSION):
                log("ALL_TASKS_COMPLETE")
            else:
                log(f"TASK_ADVANCE {task} -> {TASK_PROGRESSION[state['task_idx']]}")
        save_overnight_state(state)
        return "plateau-skip"

    # 4. Bake
    bake_candidate(head_name, pick)
    state["total_bakes"] += 1
    save_overnight_state(state)

    baked_new = load_head_neurons(head_name)
    new_val = ensemble_accuracy_c19(baked_new, val[0], val[1], n_in)
    new_test = ensemble_accuracy_c19(baked_new, test[0], test[1], n_in)

    p_str = fmt_parents(pick["parents"], n_in)
    w_str = fmt_weights(pick["weights"])
    log(
        f"BAKE {head_name} N{len(baked)}: {p_str} {w_str} c={pick['c']:.2f} rho={pick['rho']:.2f} "
        f"dV={pick['delta']:+.1f}pp -> val={new_val:.1f}% test={new_test:.1f}%  ({t_search:.1f}s search)"
    )

    # Auto-advance if now at 100%
    if new_val >= 100.0 - 1e-6:
        state["bit_idx"] += 1
        state["total_advances"] += 1
        if state["bit_idx"] >= 9:
            state["task_idx"] += 1
            state["bit_idx"] = 0
            if state["task_idx"] >= len(TASK_PROGRESSION):
                log("ALL_TASKS_COMPLETE")
            else:
                log(f"TASK_ADVANCE {task} -> {TASK_PROGRESSION[state['task_idx']]}")
        else:
            log(f"BIT_ADVANCE {head_name} -> bit {state['bit_idx']}")
        save_overnight_state(state)

    return "baked"


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "step"
    if action == "status":
        state = load_overnight_state()
        print(json.dumps(state, indent=2))
        # Summary
        bakes = state.get("total_bakes", 0)
        adv = state.get("total_advances", 0)
        task_idx = state.get("task_idx", 0)
        bit_idx = state.get("bit_idx", 0)
        task = TASK_PROGRESSION[task_idx] if task_idx < len(TASK_PROGRESSION) else "DONE"
        print(f"position: {task} bit {bit_idx}  |  bakes: {bakes}  |  advances: {adv}")
    elif action == "step":
        result = step()
        print(f"action: {result}")
    elif action == "reset":
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("state reset")
    elif action == "seed-from-existing":
        # Advance past any heads already at 100% from previous sessions
        state = load_overnight_state()
        advanced = 0
        while state["task_idx"] < len(TASK_PROGRESSION):
            head_name = current_head_name(state)
            baked = load_head_neurons(head_name)
            if not baked:
                break
            (_, val, _, n_in) = gen_data(head_name, SEED)
            cur_val = ensemble_accuracy_c19(baked, val[0], val[1], n_in)
            if cur_val >= 100.0 - 1e-6:
                state["bit_idx"] += 1
                state["total_advances"] += 1
                advanced += 1
                if state["bit_idx"] >= 9:
                    state["task_idx"] += 1
                    state["bit_idx"] = 0
                    if state["task_idx"] >= len(TASK_PROGRESSION):
                        break
            else:
                break
        save_overnight_state(state)
        task = TASK_PROGRESSION[state["task_idx"]] if state["task_idx"] < len(TASK_PROGRESSION) else "DONE"
        print(f"seeded past {advanced} existing heads -> {task} bit {state['bit_idx']}")
    else:
        print(f"unknown action: {action}")
        sys.exit(2)


if __name__ == "__main__":
    main()
