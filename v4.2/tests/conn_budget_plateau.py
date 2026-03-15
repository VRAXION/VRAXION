"""Connection budget sweep with plateau detection.

NOT fixed N steps — runs until accuracy AND density both plateau.
Plateau = no improvement in last STALE_WINDOW attempts.

Tests: uncapped vs various budgets.
Tracks density trajectory over time.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg
from scipy import sparse

SEED = 42  # single seed first, deterministic
MAX_ATTEMPTS = 200000  # hard safety cap
STALE_WINDOW = 12000   # plateau: no improvement in this many attempts
CHECK_INTERVAL = 1000  # log density/acc every N attempts

NET_CONFIGS = [
    ("V64_N192",  64, 192, 0.06, 0.5),
    ("V128_N384", 128, 384, 0.06, 0.5),
]

# Budget modes: 0 = uncapped
# Budget expressed as fraction of N*N for comparability
BUDGET_MODES = [
    ("uncapped",  0),
    ("cap_2pct",  0.02),
    ("cap_4pct",  0.04),
    ("cap_6pct",  0.06),
    ("cap_10pct", 0.10),
    ("cap_15pct", 0.15),
]


def run_one(net_name, V, N, density, threshold, budget_name, budget_frac, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)

    conn_budget = int(budget_frac * N * N) if budget_frac > 0 else 0
    net = SelfWiringGraph(N, V, density=density, threshold=threshold,
                          conn_budget=conn_budget)
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
    stale = 0
    trajectory = []
    t0 = time.perf_counter()

    for att in range(MAX_ATTEMPTS):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z; lk_s = net.leak

        net.mutate_with_mood()

        a, s = eval_b()
        if s > score:
            score = s; best_acc = max(best_acc, a)
            stale = 0
        else:
            net.mask = sm; net.mood_x = mx_s; net.mood_z = mz_s; net.leak = lk_s
            stale += 1

        # Periodic checkpoint
        if (att + 1) % CHECK_INTERVAL == 0:
            conns = int((net.mask != 0).sum())
            elapsed = time.perf_counter() - t0
            dens = conns / (N * N - N)
            trajectory.append({
                'att': att + 1, 'acc': best_acc, 'conns': conns,
                'density': dens, 'leak': net.leak, 'stale': stale,
                'elapsed': elapsed,
            })
            log_msg(log_q, f"{net_name:12s} {budget_name:10s} "
                    f"att={att+1:6d} acc={best_acc*100:5.1f}% "
                    f"conns={conns:5d} d={dens:.3f} "
                    f"leak={net.leak:.3f} stale={stale:5d}")

        # Plateau detection
        if stale >= STALE_WINDOW:
            conns = int((net.mask != 0).sum())
            elapsed = time.perf_counter() - t0
            dens = conns / (N * N - N)
            log_msg(log_q, f"{net_name:12s} {budget_name:10s} "
                    f"PLATEAU at att={att+1} acc={best_acc*100:.1f}% "
                    f"conns={conns} d={dens:.3f} "
                    f"leak={net.leak:.3f} elapsed={elapsed:.0f}s")
            break

    final_conns = int((net.mask != 0).sum())
    final_density = final_conns / (N * N - N)
    total_time = time.perf_counter() - t0

    return {
        'net': net_name, 'budget': budget_name, 'budget_frac': budget_frac,
        'conn_budget': conn_budget, 'seed': seed,
        'acc': best_acc, 'leak': net.leak,
        'init_conns': init_conns, 'final_conns': final_conns,
        'init_density': density, 'final_density': final_density,
        'total_att': att + 1, 'total_time': total_time,
        'plateau': stale >= STALE_WINDOW,
        'trajectory': trajectory,
    }


def main():
    print(f"CONNECTION BUDGET PLATEAU SWEEP", flush=True)
    print(f"Plateau detection: stale_window={STALE_WINDOW}, max={MAX_ATTEMPTS}", flush=True)
    print(f"Single seed={SEED}, sequential (one by one, plateau monitored)", flush=True)
    print("=" * 100, flush=True)

    all_results = []
    with live_log('conn_budget_plateau') as (log_q, log_path):
        for net_name, V, N, d, th in NET_CONFIGS:
            for budget_name, budget_frac in BUDGET_MODES:
                log_msg(log_q, f"--- Starting {net_name} {budget_name} ---")
                result = run_one(net_name, V, N, d, th,
                                budget_name, budget_frac, SEED, log_q)
                all_results.append(result)

                status = "PLATEAU" if result['plateau'] else f"MAX({MAX_ATTEMPTS})"
                print(f"  {net_name:12s} {budget_name:10s}: "
                      f"acc={result['acc']*100:5.1f}% "
                      f"att={result['total_att']:6d} ({status}) "
                      f"conns={result['init_conns']}->{result['final_conns']} "
                      f"density={result['init_density']:.3f}->{result['final_density']:.3f} "
                      f"leak={result['leak']:.3f} "
                      f"time={result['total_time']:.0f}s",
                      flush=True)

    # Summary
    print(f"\n{'='*100}", flush=True)
    print("SUMMARY:", flush=True)
    for net_name, V, N, d, th in NET_CONFIGS:
        print(f"\n  {net_name}:", flush=True)
        print(f"    {'budget':>10s} {'acc':>6s} {'attempts':>8s} {'conns':>8s} "
              f"{'density':>8s} {'leak':>6s} {'time':>6s} {'status':>8s}", flush=True)
        for r in all_results:
            if r['net'] == net_name:
                status = "PLATEAU" if r['plateau'] else "MAX"
                print(f"    {r['budget']:>10s} {r['acc']*100:5.1f}% {r['total_att']:8d} "
                      f"{r['final_conns']:8d} {r['final_density']:8.3f} "
                      f"{r['leak']:6.3f} {r['total_time']:5.0f}s {status:>8s}",
                      flush=True)

    # Density trajectories for each config
    print(f"\nDENSITY TRAJECTORIES (every {CHECK_INTERVAL} att):", flush=True)
    for r in all_results:
        traj = r['trajectory']
        if len(traj) >= 3:
            start = traj[0]
            mid = traj[len(traj)//2]
            end = traj[-1]
            print(f"  {r['net']:12s} {r['budget']:10s}: "
                  f"d@{start['att']}={start['density']:.3f} "
                  f"d@{mid['att']}={mid['density']:.3f} "
                  f"d@{end['att']}={end['density']:.3f} "
                  f"(slope last 3K: {(end['density']-traj[-4]['density']) if len(traj)>=4 else 0:.4f})",
                  flush=True)

    print(f"\n{'='*100}", flush=True)


if __name__ == '__main__':
    main()
