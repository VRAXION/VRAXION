"""
Sparse vs Brute Force Test — v22 SelfWiringGraph
==================================================
At small V (16-64), brute force fills to 98% density and wins.
At large V (128+), the search space EXPLODES (N=256 -> 65K connections).
Brute force can't explore it all in 16K attempts.

QUESTION: Does intelligent sparse search beat brute force at scale?

Modes:
  1. brute_force    -- no density cap, standard mutation (what won at V=64)
  2. sparse_temp    -- density cap 20% + temperature mutation
  3. sparse_cap30   -- density cap 30% + standard mutation
  4. sparse_grow    -- start sparse (6%), grow only on improvement

Test: V=128, internal=128 (N=256), 16K attempts, 3 seeds
Uses batch forward for speed (17x+ at V=128).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from v22_best_config import SelfWiringGraph, softmax


# ============================================================
#  Batch forward (from batch_forward_test.py)
# ============================================================

def forward_batch_nopriming(net, V, ticks=8):
    """All V inputs at once, no warmup. (V,N)@(N,N) matmul."""
    Weff = net.W * net.mask
    N = net.N
    worlds = np.eye(V, dtype=np.float32)
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = worlds
        raw = acts @ Weff + acts * 0.1
        charges += raw * 0.3
        charges *= net.leak
        acts = np.maximum(charges - net.threshold, 0.0)
        charges = np.clip(charges, -net.threshold * 2, net.threshold * 2)
    return charges[:, :V]


def score_batch(net, targets, V, ticks=8):
    """Batch scoring: combined 0.5*acc + 0.5*target_prob."""
    logits_all = forward_batch_nopriming(net, V, ticks)
    e = np.exp(logits_all - logits_all.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs_all, axis=1)
    acc = (preds == targets).mean()
    target_probs = probs_all[np.arange(V), targets]
    score = 0.5 * (preds == targets).astype(float).mean() + 0.5 * target_probs.mean()
    return score, acc


# ============================================================
#  Mutation helpers
# ============================================================

def flip_single(net):
    """Flip sign of one random existing connection."""
    alive = np.argwhere(net.mask != 0)
    if len(alive) == 0:
        return
    idx = alive[random.randint(0, len(alive) - 1)]
    net.mask[int(idx[0]), int(idx[1])] *= -1


def capped_mutate(net, cap, rate=0.03):
    """Mutation that respects density cap. No add if over cap."""
    N = net.N
    density = (net.mask != 0).sum() / (N * (N - 1))

    if density >= cap:
        r = random.random()
        if r < net.flip_rate:
            flip_single(net)
        else:
            action = random.choice(['remove', 'rewire'])
            alive = np.argwhere(net.mask != 0)
            if action == 'remove' and len(alive) > 3:
                idx = alive[random.randint(0, len(alive) - 1)]
                net.mask[int(idx[0]), int(idx[1])] = 0
            elif action == 'rewire' and len(alive) > 0:
                idx = alive[random.randint(0, len(alive) - 1)]
                r2, c = int(idx[0]), int(idx[1])
                old_sign = net.mask[r2, c]
                old_w = net.W[r2, c]
                net.mask[r2, c] = 0
                nc = random.randint(0, N - 1)
                while nc == r2:
                    nc = random.randint(0, N - 1)
                net.mask[r2, nc] = old_sign
                net.W[r2, nc] = old_w
    else:
        net.mutate_structure(rate)


def temp_mutate(net, temperature, cap=0.20):
    """Temperature-modulated mutation with density cap."""
    if temperature < 0.5:
        flip_single(net)
    elif temperature < 1.5:
        if random.random() < 0.3:
            flip_single(net)
        else:
            capped_mutate(net, cap, 0.05)
    elif temperature < 3.0:
        n_changes = int(2 + temperature)
        for _ in range(n_changes):
            if random.random() < 0.5:
                flip_single(net)
            else:
                capped_mutate(net, cap, 0.03)
    else:
        n_changes = int(temperature * 2)
        for _ in range(n_changes):
            capped_mutate(net, cap, 0.03)
        N = net.N
        region = random.sample(range(N), min(10, N))
        for n in region:
            alive_row = np.argwhere(net.mask[n] != 0).flatten()
            if len(alive_row) > 0:
                idx = alive_row[random.randint(0, len(alive_row) - 1)]
                net.mask[n, idx] *= -1


# ============================================================
#  Training functions (one per mode)
# ============================================================

def train_brute_force(args):
    """Mode 1: No density cap, standard mutation. (What won at V=64.)"""
    V = args['V']
    internal = args['internal']
    seed = args['seed']
    max_att = args['max_att']

    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_batch(net, perm, V)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.time()

    for att in range(max_att):
        state = net.save_state()
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        new_score, new_acc = score_batch(net, perm, V)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.time() - t0
    density = (net.mask != 0).sum() / (N * (N - 1))
    return {
        'mode': 'brute_force',
        'seed': seed, 'V': V, 'N': N,
        'acc': best_acc, 'kept': kept, 'attempts': att + 1,
        'conns': net.count_connections(), 'density': density,
        'time': elapsed,
    }


def train_sparse_temp(args):
    """Mode 2: Density cap 20% + temperature mutation."""
    V = args['V']
    internal = args['internal']
    seed = args['seed']
    max_att = args['max_att']
    cap = args.get('cap', 0.20)

    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_batch(net, perm, V)
    best_acc = acc
    kept = 0
    stale = 0
    temperature = 1.0
    t0 = time.time()

    for att in range(max_att):
        state = net.save_state()
        temp_mutate(net, temperature, cap)

        new_score, new_acc = score_batch(net, perm, V)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
            temperature = max(0.2, temperature * 0.95)
        else:
            net.restore_state(state)
            stale += 1
            if stale % 150 == 0:
                temperature = min(8.0, temperature * 1.3)

        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.time() - t0
    density = (net.mask != 0).sum() / (N * (N - 1))
    return {
        'mode': f'sparse_temp_{int(cap*100)}%',
        'seed': seed, 'V': V, 'N': N,
        'acc': best_acc, 'kept': kept, 'attempts': att + 1,
        'conns': net.count_connections(), 'density': density,
        'time': elapsed, 'final_temp': temperature,
    }


def train_sparse_cap(args):
    """Mode 3: Density cap + standard phase transition (no temperature)."""
    V = args['V']
    internal = args['internal']
    seed = args['seed']
    max_att = args['max_att']
    cap = args.get('cap', 0.30)

    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_batch(net, perm, V)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.time()

    for att in range(max_att):
        state = net.save_state()
        if phase == "STRUCTURE":
            capped_mutate(net, cap, 0.05)
        else:
            if random.random() < 0.3:
                capped_mutate(net, cap, 0.02)
            else:
                net.mutate_weights()

        new_score, new_acc = score_batch(net, perm, V)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.time() - t0
    density = (net.mask != 0).sum() / (N * (N - 1))
    return {
        'mode': f'sparse_cap_{int(cap*100)}%',
        'seed': seed, 'V': V, 'N': N,
        'acc': best_acc, 'kept': kept, 'attempts': att + 1,
        'conns': net.count_connections(), 'density': density,
        'time': elapsed,
    }


def train_sparse_grow(args):
    """Mode 4: Start sparse, grow ONLY on improvement.
    Start at 6% density. Add connections only when a mutation improves score.
    Otherwise: flip/rewire existing connections (never add blind).
    """
    V = args['V']
    internal = args['internal']
    seed = args['seed']
    max_att = args['max_att']

    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_batch(net, perm, V)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.time()

    for att in range(max_att):
        state = net.save_state()

        if phase == "STRUCTURE":
            r = random.random()
            if r < 0.4:
                # Flip existing (most powerful)
                flip_single(net)
            elif r < 0.7:
                # Rewire (move connection to new target)
                alive = np.argwhere(net.mask != 0)
                if len(alive) > 0:
                    idx = alive[random.randint(0, len(alive) - 1)]
                    r2, c = int(idx[0]), int(idx[1])
                    old_sign = net.mask[r2, c]
                    old_w = net.W[r2, c]
                    net.mask[r2, c] = 0
                    nc = random.randint(0, N - 1)
                    while nc == r2:
                        nc = random.randint(0, N - 1)
                    net.mask[r2, nc] = old_sign
                    net.W[r2, nc] = old_w
            else:
                # Targeted add: add ONE new connection
                dead = np.argwhere(net.mask == 0)
                dead = dead[dead[:, 0] != dead[:, 1]]
                if len(dead) > 0:
                    idx = dead[random.randint(0, len(dead) - 1)]
                    r2, c = int(idx[0]), int(idx[1])
                    net.mask[r2, c] = random.choice([-1.0, 1.0])
                    net.W[r2, c] = random.choice([np.float32(0.5), np.float32(1.5)])
        else:
            if random.random() < 0.4:
                flip_single(net)
            elif random.random() < 0.7:
                net.mutate_weights()
            else:
                # Add ONE connection
                dead = np.argwhere(net.mask == 0)
                dead = dead[dead[:, 0] != dead[:, 1]]
                if len(dead) > 0:
                    idx = dead[random.randint(0, len(dead) - 1)]
                    r2, c = int(idx[0]), int(idx[1])
                    net.mask[r2, c] = random.choice([-1.0, 1.0])
                    net.W[r2, c] = random.choice([np.float32(0.5), np.float32(1.5)])

        new_score, new_acc = score_batch(net, perm, V)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.time() - t0
    density = (net.mask != 0).sum() / (N * (N - 1))
    return {
        'mode': 'sparse_grow',
        'seed': seed, 'V': V, 'N': N,
        'acc': best_acc, 'kept': kept, 'attempts': att + 1,
        'conns': net.count_connections(), 'density': density,
        'time': elapsed,
    }


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777]

if __name__ == "__main__":
    n_workers = min(os.cpu_count() or 12, 24)

    print(f"  Sparse vs Brute Force Test — v22 SelfWiringGraph")
    print(f"  {'='*60}")
    print(f"  Question: At V=128+, does sparse intelligent search")
    print(f"            beat brute force density filling?")
    print(f"  CPU cores: {n_workers}")
    print(f"  Seeds: {SEEDS}")

    # =========================================================
    # PHASE A: V=128, internal=128 (N=256), 16K attempts
    # =========================================================
    print(f"\n{'#'*65}")
    print(f"  PHASE A: V=128, N=256, 16K attempts")
    print(f"  Search space: 65K possible connections")
    print(f"  5 modes x 3 seeds = 15 jobs")
    print(f"{'#'*65}", flush=True)

    V, internal, max_att = 128, 128, 16000

    jobs = []
    base_args = {'V': V, 'internal': internal, 'max_att': max_att}

    # Mode 1: Brute force (no cap)
    for seed in SEEDS:
        jobs.append(('brute', {**base_args, 'seed': seed}))

    # Mode 2: Sparse + temperature (cap 20%)
    for seed in SEEDS:
        jobs.append(('temp20', {**base_args, 'seed': seed, 'cap': 0.20}))

    # Mode 3: Sparse cap 30% (no temperature)
    for seed in SEEDS:
        jobs.append(('cap30', {**base_args, 'seed': seed, 'cap': 0.30}))

    # Mode 4: Sparse grow (start 6%, grow on improvement)
    for seed in SEEDS:
        jobs.append(('grow', {**base_args, 'seed': seed}))

    # Mode 5: Sparse + temperature (cap 40%) -- higher cap
    for seed in SEEDS:
        jobs.append(('temp40', {**base_args, 'seed': seed, 'cap': 0.40}))

    # Map job type to function
    fn_map = {
        'brute': train_brute_force,
        'temp20': train_sparse_temp,
        'cap30': train_sparse_cap,
        'grow': train_sparse_grow,
        'temp40': train_sparse_temp,
    }

    print(f"\n  Running {len(jobs)} jobs...", flush=True)
    t0 = time.time()

    # Submit all jobs
    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for jtype, jargs in jobs:
            futures.append(pool.submit(fn_map[jtype], jargs))
        results_a = [f.result() for f in futures]

    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    # Aggregate
    groups = defaultdict(list)
    for r in results_a:
        groups[r['mode']].append(r)

    modes_order = ['brute_force', 'sparse_temp_20%', 'sparse_cap_30%',
                   'sparse_grow', 'sparse_temp_40%']

    print(f"\n  {'Mode':<22s} {'Acc':>7s} {'Conns':>7s} {'Density':>9s} "
          f"{'Kept':>6s} {'Att':>7s} {'Time':>6s}")
    print(f"  {'-'*70}")
    for mode in modes_order:
        runs = groups.get(mode, [])
        if not runs:
            continue
        accs = [r['acc'] for r in runs]
        conns = [r['conns'] for r in runs]
        densities = [r['density'] for r in runs]
        kepts = [r['kept'] for r in runs]
        atts = [r['attempts'] for r in runs]
        times = [r['time'] for r in runs]
        print(f"  {mode:<22s} {np.mean(accs)*100:5.1f}% {np.mean(conns):5.0f} "
              f"{np.mean(densities)*100:7.1f}% {np.mean(kepts):5.0f} "
              f"{np.mean(atts):5.0f} {np.mean(times):5.0f}s")

    # Per-seed detail
    print(f"\n  Per-seed detail:")
    print(f"  {'Mode':<22s} {'Seed':>5s} {'Acc':>7s} {'Conns':>7s} {'Density':>9s}")
    print(f"  {'-'*55}")
    for mode in modes_order:
        runs = groups.get(mode, [])
        for r in sorted(runs, key=lambda x: x['seed']):
            print(f"  {mode:<22s} {r['seed']:>5d} {r['acc']*100:5.1f}% "
                  f"{r['conns']:>5d} {r['density']*100:7.1f}%")

    # =========================================================
    # PHASE B: Bigger -- V=192, N=384 (if Phase A under 3 min)
    # =========================================================
    phase_a_time = time.time() - t0
    if phase_a_time < 180:
        print(f"\n{'#'*65}")
        print(f"  PHASE B: V=192, N=384, 16K attempts")
        print(f"  Search space: 147K possible connections")
        print(f"  Top 2 modes from Phase A + brute force, 2 seeds")
        print(f"{'#'*65}", flush=True)

        # Pick top 2 non-brute modes
        mode_accs = {}
        for mode in modes_order:
            runs = groups.get(mode, [])
            if runs:
                mode_accs[mode] = np.mean([r['acc'] for r in runs])

        ranked = sorted(mode_accs.items(), key=lambda x: -x[1])
        top_modes = ['brute_force']
        for mode, _ in ranked:
            if mode != 'brute_force' and len(top_modes) < 3:
                top_modes.append(mode)

        V2, internal2, max_att2 = 192, 192, 16000
        base_args2 = {'V': V2, 'internal': internal2, 'max_att': max_att2}
        seeds2 = [42, 123]

        jobs2 = []
        for mode in top_modes:
            for seed in seeds2:
                if mode == 'brute_force':
                    jobs2.append(('brute', {**base_args2, 'seed': seed}))
                elif 'temp_20' in mode:
                    jobs2.append(('temp20', {**base_args2, 'seed': seed, 'cap': 0.20}))
                elif 'temp_40' in mode:
                    jobs2.append(('temp40', {**base_args2, 'seed': seed, 'cap': 0.40}))
                elif 'cap_30' in mode:
                    jobs2.append(('cap30', {**base_args2, 'seed': seed, 'cap': 0.30}))
                elif mode == 'sparse_grow':
                    jobs2.append(('grow', {**base_args2, 'seed': seed}))

        print(f"\n  Running {len(jobs2)} jobs ({top_modes})...", flush=True)
        t1 = time.time()

        futures2 = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for jtype, jargs in jobs2:
                futures2.append(pool.submit(fn_map[jtype], jargs))
            results_b = [f.result() for f in futures2]

        print(f"  Completed in {time.time() - t1:.0f}s", flush=True)

        groups_b = defaultdict(list)
        for r in results_b:
            groups_b[r['mode']].append(r)

        print(f"\n  {'Mode':<22s} {'Acc':>7s} {'Conns':>7s} {'Density':>9s} "
              f"{'Kept':>6s} {'Time':>6s}")
        print(f"  {'-'*60}")
        for mode in top_modes:
            # Map to actual mode name
            mode_key = mode
            runs = groups_b.get(mode_key, [])
            if not runs:
                continue
            accs = [r['acc'] for r in runs]
            conns = [r['conns'] for r in runs]
            densities = [r['density'] for r in runs]
            kepts = [r['kept'] for r in runs]
            times = [r['time'] for r in runs]
            print(f"  {mode_key:<22s} {np.mean(accs)*100:5.1f}% {np.mean(conns):5.0f} "
                  f"{np.mean(densities)*100:7.1f}% {np.mean(kepts):5.0f} "
                  f"{np.mean(times):5.0f}s")
    else:
        print(f"\n  Phase A took {phase_a_time:.0f}s > 180s, skipping Phase B")

    # =========================================================
    # VERDICT
    # =========================================================
    print(f"\n  {'='*65}")
    print(f"  VERDICT")
    print(f"  {'='*65}")

    mode_accs = {}
    for mode in modes_order:
        runs = groups.get(mode, [])
        if runs:
            mode_accs[mode] = np.mean([r['acc'] for r in runs])

    best_mode = max(mode_accs.items(), key=lambda x: x[1])
    brute_acc = mode_accs.get('brute_force', 0)

    print(f"\n  V=128 (N=256):")
    for rank, (mode, acc) in enumerate(sorted(mode_accs.items(),
                                               key=lambda x: -x[1]), 1):
        marker = " <<<" if rank == 1 else ""
        dens = np.mean([r['density'] for r in groups.get(mode, [])])
        print(f"    #{rank} {mode:<22s} acc={acc*100:5.1f}% density={dens*100:.0f}%{marker}")

    if best_mode[0] == 'brute_force':
        print(f"\n  -> BRUTE FORCE STILL WINS at V=128")
        print(f"     The search space isn't big enough to punish density filling")
    else:
        diff = (best_mode[1] - brute_acc) * 100
        print(f"\n  -> SPARSE WINS! {best_mode[0]} beats brute force by {diff:.1f}%")
        print(f"     Intelligent search matters when the space is too large")

    print(f"\n  {'='*65}")
    print(f"  DONE")
    print(f"  {'='*65}", flush=True)
