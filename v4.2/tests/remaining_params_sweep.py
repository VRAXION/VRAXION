"""Wide+shallow sweep of all remaining untested params.

threshold, ticks, clip_factor, mood step/prob.
V64 only (fast), 3 seeds, 16K budget (half normal — just to see the landscape).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg
from scipy import sparse

SEEDS = [42, 77, 123]
BUDGET = 16000
V, N, DENSITY = 64, 192, 0.06


def run_one(sweep_name, param_label, threshold, ticks, clip_factor,
            mood_step, mood_prob, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY, threshold=threshold)
    net.clip_factor = clip_factor
    perm = np.random.permutation(V)

    def eval_b():
        Weff_csr = sparse.csr_matrix(net.mask.astype(np.float32) * net.gain)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(ticks):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = np.asarray(acts @ Weff_csr) + acts * net.self_conn
            np.nan_to_num(raw, copy=False)
            charges += raw * net.charge_rate
            charges *= np.float32(net.leak)
            acts = np.maximum(charges - net.threshold, 0)
            charges = np.clip(charges, -clip_bound, clip_bound)
        out = charges[:, net.out_start:net.out_start + V]
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return acc, 0.5 * acc + 0.5 * tp

    # Custom mood mutation with variable step/prob
    def mutate_custom():
        if random.random() < mood_prob:
            net.mood_x = np.clip(net.mood_x + random.gauss(0, mood_step), 0.0, 1.0)
        if random.random() < mood_prob:
            net.mood_z = np.clip(net.mood_z + random.gauss(0, mood_step), 0.0, 1.0)
        if random.random() < 0.2:
            net.leak = np.clip(net.leak + random.gauss(0, 0.03), 0.5, 0.99)
        n_changes = max(1, int(1 + net.mood_z * 14))
        for _ in range(n_changes):
            if net.mood_x < 0.25:
                if random.random() < 0.7: net._add_connection()
                else: net._flip_connection()
            elif net.mood_x < 0.50:
                r = random.random()
                if r < 0.6: net._rewire_connection()
                elif r < 0.8: net._flip_connection()
                else: net._add_connection()
            elif net.mood_x < 0.75:
                if random.random() < 0.8: net._flip_connection()
                else: net._rewire_connection()
            else:
                r = random.random()
                if r < 0.7: net._remove_connection()
                elif r < 0.9: net._flip_connection()
                else: net._rewire_connection()
        net._weff_dirty = True

    _, score = eval_b()
    best_acc = 0.0
    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z; lk_s = net.leak
        mutate_custom()
        a, s = eval_b()
        if s > score:
            score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.mood_x = mx_s; net.mood_z = mz_s; net.leak = lk_s

    log_msg(log_q, f"{sweep_name:12s} {param_label:16s} seed={seed:3d} "
            f"acc={best_acc*100:5.1f}% leak={net.leak:.3f}")
    return {
        'sweep': sweep_name, 'param': param_label, 'seed': seed,
        'acc': best_acc, 'leak': net.leak,
    }


def main():
    # Default values for reference
    DEF_TH = 0.5; DEF_TICKS = 8; DEF_CF = 2.0
    DEF_MSTEP = 0.15; DEF_MPROB = 0.2

    jobs = []

    # 1. Threshold sweep
    for th in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]:
        label = f"th={th:.2f}"
        for seed in SEEDS:
            jobs.append(("threshold", label, th, DEF_TICKS, DEF_CF, DEF_MSTEP, DEF_MPROB, seed))

    # 2. Ticks sweep
    for ticks in [2, 4, 6, 8, 12, 16]:
        label = f"ticks={ticks}"
        for seed in SEEDS:
            jobs.append(("ticks", label, DEF_TH, ticks, DEF_CF, DEF_MSTEP, DEF_MPROB, seed))

    # 3. Clip factor sweep
    for cf in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        label = f"cf={cf:.1f}"
        for seed in SEEDS:
            jobs.append(("clip_factor", label, DEF_TH, DEF_TICKS, cf, DEF_MSTEP, DEF_MPROB, seed))

    # 4. Mood step sweep
    for ms in [0.05, 0.10, 0.15, 0.25, 0.40]:
        label = f"mstep={ms:.2f}"
        for seed in SEEDS:
            jobs.append(("mood_step", label, DEF_TH, DEF_TICKS, DEF_CF, ms, DEF_MPROB, seed))

    # 5. Mood probability sweep
    for mp in [0.05, 0.10, 0.20, 0.35, 0.50]:
        label = f"mprob={mp:.2f}"
        for seed in SEEDS:
            jobs.append(("mood_prob", label, DEF_TH, DEF_TICKS, DEF_CF, DEF_MSTEP, mp, seed))

    total = len(jobs)
    print(f"REMAINING PARAMS WIDE SWEEP: {total} jobs, 22 workers, {BUDGET} budget", flush=True)
    print(f"V={V} N={N} density={DENSITY}", flush=True)
    print("=" * 85, flush=True)

    all_results = []
    with live_log('remaining_params') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=22) as pool:
            futures = [pool.submit(run_one, *j, log_q) for j in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary per sweep
    print(f"\n{'='*85}", flush=True)
    for sweep_name in ["threshold", "ticks", "clip_factor", "mood_step", "mood_prob"]:
        print(f"\n{sweep_name.upper()}:", flush=True)
        params = sorted(set(r['param'] for r in all_results if r['sweep']==sweep_name))
        for p in params:
            runs = [r['acc'] for r in all_results if r['sweep']==sweep_name and r['param']==p]
            mean = np.mean(runs) * 100
            std = np.std(runs) * 100
            marker = " <<<" if mean >= max(np.mean([r['acc'] for r in all_results
                        if r['sweep']==sweep_name and r['param']==pp]) for pp in params) * 100 - 0.1 else ""
            print(f"  {p:16s} {mean:5.1f}% +/-{std:.1f}%{marker}", flush=True)

    print(f"\n{'='*85}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
