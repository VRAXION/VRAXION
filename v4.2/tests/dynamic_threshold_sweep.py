"""Dynamic threshold weight: act < C -> weak, act >= C -> strong. Sweep C."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg

V = 64; N = 192; BUDGET = 16000; SEEDS = [42, 77, 123]
WEAK = 1.5; STRONG = 2.0

C_VALUES = [0.1, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def run_one(mode, C, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    def eval_b():
        mask_f = net.mask.astype(np.float32)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            if mode == 'static':
                Weff = (WEAK + net.W_strong.astype(np.float32) * (STRONG - WEAK)) * mask_f
            else:
                # Dynamic: per-neuron weight based on activation vs C
                # For batch: acts is (V, N), we need per-neuron decision
                # Use max activation across batch as proxy
                neuron_act = np.abs(acts).max(axis=0)  # (N,) max act per neuron
                is_strong = neuron_act >= C
                row_weight = np.where(is_strong, np.float32(STRONG), np.float32(WEAK))
                Weff = mask_f * row_weight[:, None]
            raw = acts @ Weff + acts * net.self_conn
            np.nan_to_num(raw, copy=False)
            charges += raw * net.charge_rate
            charges *= net.leak
            acts = np.maximum(charges - net.threshold, 0)
            charges = np.clip(charges, -clip_bound, clip_bound)
        out = charges[:, net.out_start:net.out_start + V]
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return acc, 0.5*acc + 0.5*tp

    _, score = eval_b()
    best_acc = 0.0
    for att in range(BUDGET):
        sm = net.mask.copy()
        sw = net.W_strong.copy() if mode == 'static' else None
        mx_s = net.mood_x; mz_s = net.mood_z

        # Mood mutation - only mask for dynamic, mask+weight for static
        net.mutate_with_mood()

        a, s = eval_b()
        if s > score:
            score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm
            if mode == 'static':
                net.W_strong = sw
            net.mood_x = mx_s; net.mood_z = mz_s

    label = f"{mode}_C={C:.2f}" if mode == 'dynamic' else 'static_bool'
    log_msg(log_q, f"{label:20s} seed={seed:3d} acc={best_acc*100:5.1f}%")
    return {'mode': mode, 'C': C, 'seed': seed, 'acc': best_acc}


def main():
    # Static baseline + dynamic C sweep
    jobs = []
    for seed in SEEDS:
        jobs.append(('static', 0, seed))
    for C in C_VALUES:
        for seed in SEEDS:
            jobs.append(('dynamic', C, seed))

    total = len(jobs)
    print(f"DYNAMIC THRESHOLD SWEEP: {total} jobs (1 static + {len(C_VALUES)} C values x {len(SEEDS)} seeds)", flush=True)
    print(f"C values: {C_VALUES}", flush=True)
    print(f"weak={WEAK}, strong={STRONG}", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    with live_log('dynamic_threshold') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(run_one, m, c, s, log_q) for m, c, s in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary
    groups = defaultdict(list)
    for r in all_results:
        key = f"{r['mode']}_C={r['C']:.2f}" if r['mode'] == 'dynamic' else 'static_bool'
        groups[key].append(r['acc'])

    print(f"\n{'='*70}", flush=True)
    print("RESULTS:", flush=True)
    ranked = [(k, np.mean(v)*100, np.std(v)*100) for k, v in groups.items()]
    ranked.sort(key=lambda x: -x[1])
    for name, mean, std in ranked:
        marker = ' <<<' if name == ranked[0][0] else ''
        print(f"  {name:20s}  acc={mean:5.1f}% +/-{std:.1f}%{marker}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
