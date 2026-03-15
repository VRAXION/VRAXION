"""A/B test: 3-zone mood (old, no pruner) vs 4-zone mood (with pruner).
Tracks accuracy AND density evolution."""
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


def mutate_3zone(net):
    """Old 3-zone mood: scout/rewirer/refiner (NO pruner)."""
    if random.random() < 0.2:
        net.mood_x = np.clip(net.mood_x + random.gauss(0, 0.15), 0.0, 1.0)
    if random.random() < 0.2:
        net.mood_z = np.clip(net.mood_z + random.gauss(0, 0.15), 0.0, 1.0)
    if random.random() < 0.2:
        net.leak = np.clip(net.leak + random.gauss(0, 0.03), 0.5, 0.99)

    n_changes = max(1, int(1 + net.mood_z * 14))
    for _ in range(n_changes):
        if net.mood_x < 0.33:
            if random.random() < 0.7:
                net._add_connection()
            else:
                net._flip_connection()
        elif net.mood_x < 0.66:
            r = random.random()
            if r < 0.6:
                net._rewire_connection()
            elif r < 0.8:
                net._flip_connection()
            else:
                net._add_connection()
        else:
            net._flip_connection()
    net._weff_dirty = True


def run_one(net_name, V, N, density, threshold, mode, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=density, threshold=threshold)
    perm = np.random.permutation(V)

    init_conns = int((net.mask != 0).sum())

    def eval_b():
        Weff_csr = sparse.csr_matrix(net.mask.astype(np.float32) * net.gain)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
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

    _, score = eval_b()
    best_acc = 0.0

    # Track density at checkpoints
    density_history = [(0, init_conns)]

    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z; lk_s = net.leak

        if mode == '4zone':
            net.mutate_with_mood()  # new 4-zone
        else:
            mutate_3zone(net)  # old 3-zone

        a, s = eval_b()
        if s > score:
            score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.mood_x = mx_s; net.mood_z = mz_s; net.leak = lk_s

        if att % 8000 == 0 or att == BUDGET - 1:
            conns = int((net.mask != 0).sum())
            density_history.append((att, conns))

    final_conns = int((net.mask != 0).sum())
    final_density = final_conns / (N * N - N)

    log_msg(log_q, f"{net_name:12s} {mode:6s} seed={seed:3d} "
            f"acc={best_acc*100:5.1f}% leak={net.leak:.3f} "
            f"mood_x={net.mood_x:.2f} "
            f"conns={init_conns}->{final_conns} "
            f"density={density:.3f}->{final_density:.3f}")

    return {
        'net': net_name, 'mode': mode, 'seed': seed,
        'acc': best_acc, 'leak': net.leak, 'mood_x': net.mood_x,
        'init_conns': init_conns, 'final_conns': final_conns,
        'init_density': density, 'final_density': final_density,
        'density_history': density_history,
    }


def main():
    jobs = []
    for net_name, V, N, d, th in NET_CONFIGS:
        for mode in ['3zone', '4zone']:
            for seed in SEEDS:
                jobs.append((net_name, V, N, d, th, mode, seed))

    total = len(jobs)
    print(f"PRUNER A/B TEST: {total} jobs (3zone vs 4zone), 22 workers, 32K budget", flush=True)
    print("=" * 95, flush=True)

    all_results = []
    with live_log('pruner_ab') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=22) as pool:
            futures = [pool.submit(run_one, *j, log_q) for j in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary
    nets = [c[0] for c in NET_CONFIGS]
    print(f"\n{'='*95}", flush=True)
    print("ACCURACY:", flush=True)
    print(f"  {'':12s} {'3zone':>10s} {'4zone':>10s} {'diff':>8s}", flush=True)
    for net in nets:
        old = [r['acc'] for r in all_results if r['net']==net and r['mode']=='3zone']
        new = [r['acc'] for r in all_results if r['net']==net and r['mode']=='4zone']
        if old and new:
            om, nm = np.mean(old)*100, np.mean(new)*100
            print(f"  {net:12s} {om:9.1f}% {nm:9.1f}% {nm-om:+7.1f}pp", flush=True)

    print(f"\nDENSITY EVOLUTION:", flush=True)
    print(f"  {'':12s} {'3zone_start':>12s} {'3zone_end':>10s} {'4zone_start':>12s} {'4zone_end':>10s}", flush=True)
    for net in nets:
        old = [r for r in all_results if r['net']==net and r['mode']=='3zone']
        new = [r for r in all_results if r['net']==net and r['mode']=='4zone']
        if old and new:
            oi = np.mean([r['init_density'] for r in old])
            of = np.mean([r['final_density'] for r in old])
            ni = np.mean([r['init_density'] for r in new])
            nf = np.mean([r['final_density'] for r in new])
            print(f"  {net:12s} {oi:11.3f}  {of:9.3f}  {ni:11.3f}  {nf:9.3f}", flush=True)

    print(f"\nFINAL MOOD_X (4zone only — where did it settle?):", flush=True)
    for net in nets:
        runs = [r for r in all_results if r['net']==net and r['mode']=='4zone']
        if runs:
            moods = [r['mood_x'] for r in runs]
            print(f"  {net:12s}: mood_x = {np.mean(moods):.2f} +/- {np.std(moods):.2f} "
                  f"({', '.join(f'{m:.2f}' for m in moods)})", flush=True)

    print(f"{'='*95}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
