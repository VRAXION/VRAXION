"""Auto-tuning charge_rate + density evolution tracking.

Rule: cr starts high, decreases until leak >= 0.98 (admissible).
Once stable, freezes. Compare vs fixed cr=0.3 baseline.
Also tracks: how density evolves during training.
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
BUDGET = 32000

NET_CONFIGS = [
    ("V16_N80",    16,   80, 0.06, 0.5),
    ("V64_N192",   64,  192, 0.06, 0.5),
    ("V64_dense",  64,  192, 0.15, 0.5),
    ("V64_sparse", 64,  192, 0.02, 0.5),
    ("V128_N384", 128,  384, 0.06, 0.5),
]

MODES = [
    ("fix_0.3",    0.3, False),
    ("fix_0.2",    0.2, False),
    ("autotune",   0.5, True),   # start high, auto-reduce
]


def run_one(net_name, V, N, density, threshold, mode_name, cr_init, autotune, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=density, threshold=threshold)
    perm = np.random.permutation(V)
    leak = 0.85
    cr = cr_init

    init_conns = int((net.mask != 0).sum())
    cr_frozen = False
    cr_freeze_att = -1
    cr_history = []

    def eval_b(lk, c_rate):
        Weff_csr = sparse.csr_matrix(net.mask.astype(np.float32) * net.gain)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = np.asarray(acts @ Weff_csr) + acts * net.self_conn
            np.nan_to_num(raw, copy=False)
            charges += raw * np.float32(c_rate)
            charges *= np.float32(lk)
            acts = np.maximum(charges - net.threshold, 0)
            charges = np.clip(charges, -clip_bound, clip_bound)
        out = charges[:, net.out_start:net.out_start + V]
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return acc, 0.5 * acc + 0.5 * tp

    _, score = eval_b(leak, cr)
    best_acc = 0.0

    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z
        lk_s = leak; cr_s = cr

        net.mutate_with_mood()
        if random.random() < 0.2:
            leak = np.clip(leak + random.gauss(0, 0.03), 0.5, 0.99)

        # Auto-tune cr: reduce if leak too low, freeze when stable
        if autotune and not cr_frozen:
            if att > 0 and att % 500 == 0:
                if leak < 0.98:
                    cr = max(0.05, cr * 0.85)  # reduce 15%
                elif leak >= 0.98 and att >= 2000:
                    # Leak is high and we've had enough warmup
                    cr_frozen = True
                    cr_freeze_att = att

        a, s = eval_b(leak, cr)
        if s > score:
            score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.mood_x = mx_s; net.mood_z = mz_s
            leak = lk_s; cr = cr_s

        if att % 4000 == 0 or att == BUDGET - 1:
            cr_history.append((att, cr, leak))

    final_conns = int((net.mask != 0).sum())
    final_density = final_conns / (N * N - N)  # exclude diagonal
    eff_drive = cr / (1 - leak) if leak < 1 else 999

    freeze_info = f"froze@{cr_freeze_att}" if cr_frozen else "never_froze"
    log_msg(log_q, f"{net_name:12s} {mode_name:10s} seed={seed:3d} "
            f"acc={best_acc*100:5.1f}% leak={leak:.3f} cr={cr:.3f} "
            f"drive={eff_drive:.1f} conns={init_conns}->{final_conns} "
            f"density={density:.3f}->{final_density:.3f} {freeze_info}")

    return {
        'net': net_name, 'mode': mode_name, 'seed': seed,
        'acc': best_acc, 'leak': leak, 'cr': cr,
        'drive': eff_drive,
        'init_conns': init_conns, 'final_conns': final_conns,
        'init_density': density, 'final_density': final_density,
        'cr_frozen': cr_frozen, 'cr_freeze_att': cr_freeze_att,
        'cr_history': cr_history,
    }


def main():
    jobs = []
    for net_name, V, N, d, th in NET_CONFIGS:
        for mode_name, cr_init, autotune in MODES:
            for seed in SEEDS:
                jobs.append((net_name, V, N, d, th, mode_name, cr_init, autotune, seed))

    total = len(jobs)
    print(f"CR AUTOTUNE + DENSITY EVOLUTION: {total} jobs, 22 workers, 32K budget", flush=True)
    print("=" * 95, flush=True)

    all_results = []
    with live_log('cr_autotune') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=22) as pool:
            futures = [pool.submit(run_one, *j, log_q) for j in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary tables
    nets = [c[0] for c in NET_CONFIGS]
    modes = [m[0] for m in MODES]

    print(f"\n{'='*95}", flush=True)
    print("ACCURACY:", flush=True)
    print(f"  {'':12s}", end='')
    for m in modes: print(f" {m:>12s}", end='')
    print()
    for net in nets:
        print(f"  {net:12s}", end='')
        for m in modes:
            runs = [r['acc'] for r in all_results if r['net']==net and r['mode']==m]
            if runs: print(f" {np.mean(runs)*100:11.1f}%", end='')
            else: print(f"          -- ", end='')
        print()

    print(f"\nCR FINAL VALUES (autotune):", flush=True)
    for net in nets:
        runs = [r for r in all_results if r['net']==net and r['mode']=='autotune']
        if runs:
            crs = [r['cr'] for r in runs]
            lks = [r['leak'] for r in runs]
            drives = [r['drive'] for r in runs]
            frozen = [r['cr_freeze_att'] for r in runs if r['cr_frozen']]
            print(f"  {net:12s}: cr={np.mean(crs):.3f}+/-{np.std(crs):.3f} "
                  f"leak={np.mean(lks):.3f} drive={np.mean(drives):.1f} "
                  f"froze@{frozen}", flush=True)

    print(f"\nDENSITY EVOLUTION (fix_0.3 baseline):", flush=True)
    print(f"  {'':12s} {'init':>8s} {'final':>8s} {'change':>8s} {'conns':>12s}", flush=True)
    for net in nets:
        runs = [r for r in all_results if r['net']==net and r['mode']=='fix_0.3']
        if runs:
            id_avg = np.mean([r['init_density'] for r in runs])
            fd_avg = np.mean([r['final_density'] for r in runs])
            ic = int(np.mean([r['init_conns'] for r in runs]))
            fc = int(np.mean([r['final_conns'] for r in runs]))
            print(f"  {net:12s} {id_avg:7.3f}  {fd_avg:7.3f}  {(fd_avg-id_avg)/id_avg*100:+6.1f}%  {ic:5d}->{fc:5d}",
                  flush=True)

    print(f"{'='*95}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
