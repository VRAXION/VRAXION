"""
Scaling Frontier Test -- v22 SelfWiringGraph
=============================================
Where does brute force break?

Phase A: Class Scaling — fixed 64 internal neurons, V grows
  V=16..160, N = V + 64, 16K att, 3 seeds, baseline NO CAP

Phase B: Neuron Scaling — fixed V=64, internal grows
  internal=32..512, N = 64 + internal, 16K att, 3 seeds, baseline NO CAP

Phase C: Scaling + Temperature — largest Phase B size
  baseline no cap vs baseline+30% cap vs baseline+15% cap vs temp_gradual+15% cap
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from v22_best_config import SelfWiringGraph, softmax

# ============================================================
#  Mutation helpers
# ============================================================

MAX_DENSITY_30 = 0.30
MAX_DENSITY_15 = 0.15


def flip_single(net):
    alive = np.argwhere(net.mask != 0)
    if len(alive) == 0:
        return
    idx = alive[random.randint(0, len(alive) - 1)]
    net.mask[int(idx[0]), int(idx[1])] *= -1


def capped_mutate_once(net, rate=0.03, max_density=0.15):
    N = net.N
    density = (net.mask != 0).sum() / (N * (N - 1))
    if density >= max_density:
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
                old_sign, old_w = net.mask[r2, c], net.W[r2, c]
                net.mask[r2, c] = 0
                nc = random.randint(0, N - 1)
                while nc == r2:
                    nc = random.randint(0, N - 1)
                net.mask[r2, nc] = old_sign
                net.W[r2, nc] = old_w
    else:
        net.mutate_structure(rate)


def temp_mutate_unleashed(net, temperature):
    """Temperature-modulated mutation WITHOUT density cap."""
    r = random.random()
    if temperature < 0.3:
        flip_single(net)
    elif temperature < 1.0:
        if r < 0.5:
            flip_single(net)
        else:
            net.mutate_structure(0.03)
    elif temperature < 2.0:
        if r < 0.3:
            flip_single(net)
        else:
            net.mutate_structure(0.05)
    elif temperature < 5.0:
        n_changes = int(2 + temperature)
        for _ in range(n_changes):
            if random.random() < 0.4:
                flip_single(net)
            else:
                net.mutate_structure(0.05)
    else:
        n_changes = int(temperature * 2)
        for _ in range(n_changes):
            net.mutate_structure(0.05)
        region_size = min(int(temperature * 2), net.N)
        region = random.sample(range(net.N), region_size)
        for n in region:
            alive = np.argwhere(net.mask[n] != 0).flatten()
            if len(alive) > 0:
                idx = alive[random.randint(0, len(alive) - 1)]
                net.mask[n, idx] *= -1
    if random.random() < 0.3:
        alive = np.argwhere(net.mask != 0)
        if len(alive) > 0:
            i = alive[random.randint(0, len(alive) - 1)]
            net.W[int(i[0]), int(i[1])] = (
                np.float32(1.5) if net.W[int(i[0]), int(i[1])] < 1.0
                else np.float32(0.5))


def temp_mutate_capped(net, temperature, max_density=0.15):
    """Temperature-modulated mutation WITH density cap."""
    N = net.N
    density = (net.mask != 0).sum() / (N * (N - 1))

    r = random.random()
    if temperature < 0.3:
        flip_single(net)
    elif temperature < 1.0:
        if r < 0.5:
            flip_single(net)
        else:
            capped_mutate_once(net, 0.03, max_density)
    elif temperature < 2.0:
        if r < 0.3:
            flip_single(net)
        else:
            capped_mutate_once(net, 0.05, max_density)
    elif temperature < 5.0:
        n_changes = int(2 + temperature)
        for _ in range(n_changes):
            if random.random() < 0.4:
                flip_single(net)
            else:
                capped_mutate_once(net, 0.05, max_density)
    else:
        n_changes = int(temperature * 2)
        for _ in range(n_changes):
            capped_mutate_once(net, 0.05, max_density)
        region_size = min(int(temperature * 2), net.N)
        region = random.sample(range(net.N), region_size)
        for n in region:
            alive = np.argwhere(net.mask[n] != 0).flatten()
            if len(alive) > 0:
                idx = alive[random.randint(0, len(alive) - 1)]
                net.mask[n, idx] *= -1
    if random.random() < 0.3:
        alive = np.argwhere(net.mask != 0)
        if len(alive) > 0:
            i = alive[random.randint(0, len(alive) - 1)]
            net.W[int(i[0]), int(i[1])] = (
                np.float32(1.5) if net.W[int(i[0]), int(i[1])] < 1.0
                else np.float32(0.5))


# ============================================================
#  Scoring
# ============================================================

def score_combined(net, targets, vocab, ticks=8):
    net.reset()
    correct = 0
    total_score = 0.0
    for p in range(2):
        for inp in range(vocab):
            world = np.zeros(vocab, dtype=np.float32)
            world[inp] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:vocab])
            if p == 1:
                tgt = targets[inp]
                acc_i = 1.0 if np.argmax(probs) == tgt else 0.0
                tp = float(probs[tgt])
                total_score += 0.5 * acc_i + 0.5 * tp
                if acc_i > 0:
                    correct += 1
    return total_score / vocab, correct / vocab


# ============================================================
#  Training: baseline (no cap)
# ============================================================

def train_baseline(V, internal, seed, max_attempts=16000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_combined(net, perm, V, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        new_score, new_acc = score_combined(net, perm, V, ticks)

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
        if stale >= 8000:
            break

    elapsed = time.time() - t0
    conns = net.count_connections()
    max_conns = N * (N - 1)
    density = conns / max_conns * 100 if max_conns > 0 else 0
    accept_rate = (kept / max(att + 1, 1)) * 100
    time_per_att = elapsed / max(att + 1, 1) * 1000  # ms

    return {
        'V': V, 'internal': internal, 'N': N, 'seed': seed,
        'final_acc': best_acc, 'conns': conns, 'density': density,
        'accept_rate': accept_rate, 'kept': kept,
        'attempts': att + 1, 'time': elapsed,
        'time_per_att_ms': time_per_att,
    }


# ============================================================
#  Training: baseline with density cap
# ============================================================

def train_baseline_capped(V, internal, seed, max_density=0.15,
                          max_attempts=16000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_combined(net, perm, V, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            capped_mutate_once(net, 0.05, max_density)
        else:
            if random.random() < 0.3:
                capped_mutate_once(net, 0.02, max_density)
            else:
                net.mutate_weights()

        new_score, new_acc = score_combined(net, perm, V, ticks)

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
        if stale >= 8000:
            break

    elapsed = time.time() - t0
    conns = net.count_connections()
    max_conns = N * (N - 1)
    density = conns / max_conns * 100 if max_conns > 0 else 0
    accept_rate = (kept / max(att + 1, 1)) * 100
    time_per_att = elapsed / max(att + 1, 1) * 1000

    return {
        'V': V, 'internal': internal, 'N': N, 'seed': seed,
        'final_acc': best_acc, 'conns': conns, 'density': density,
        'accept_rate': accept_rate, 'kept': kept,
        'attempts': att + 1, 'time': elapsed,
        'time_per_att_ms': time_per_att,
        'max_density': max_density,
    }


# ============================================================
#  Training: temp_gradual with density cap
# ============================================================

def train_temp_capped(V, internal, seed, max_density=0.15,
                      max_attempts=16000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_combined(net, perm, V, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    temperature = 1.0

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()
        temp_mutate_capped(net, temperature, max_density)
        new_score, new_acc = score_combined(net, perm, V, ticks)

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
            temperature = max(0.1, temperature * 0.95)
        else:
            net.restore_state(state)
            stale += 1
            if stale % 100 == 0:
                temperature = min(10.0, temperature * 1.3)

        if best_acc >= 0.99:
            break
        if stale >= 8000:
            break

    elapsed = time.time() - t0
    conns = net.count_connections()
    max_conns = N * (N - 1)
    density = conns / max_conns * 100 if max_conns > 0 else 0
    accept_rate = (kept / max(att + 1, 1)) * 100
    time_per_att = elapsed / max(att + 1, 1) * 1000

    return {
        'V': V, 'internal': internal, 'N': N, 'seed': seed,
        'final_acc': best_acc, 'conns': conns, 'density': density,
        'accept_rate': accept_rate, 'kept': kept,
        'attempts': att + 1, 'time': elapsed,
        'time_per_att_ms': time_per_att,
        'final_temp': temperature, 'max_density': max_density,
    }


# ============================================================
#  Workers
# ============================================================

def worker_baseline(args):
    try:
        return train_baseline(**args)
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc(), **args}


def worker_baseline_capped(args):
    try:
        return train_baseline_capped(**args)
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc(), **args}


def worker_temp_capped(args):
    try:
        return train_temp_capped(**args)
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc(), **args}


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777]

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24

    print(f"  Scaling Frontier Test -- v22 SelfWiringGraph")
    print(f"  {'=' * 60}")
    print(f"  Seeds: {SEEDS}")
    print(f"  CPU cores: {n_workers}")

    # ==========================================================
    # PHASE A: Class Scaling (V grows, internal=64 fixed)
    # ==========================================================
    class_configs = [16, 32, 64, 96, 128, 160]

    print(f"\n{'#' * 65}")
    print(f"  PHASE A: Class Scaling (internal=64 fix, V grows)")
    print(f"  V={class_configs}, N=V+64, 16K att, 3 seeds, baseline NO CAP")
    print(f"  {len(class_configs) * len(SEEDS)} jobs")
    print(f"{'#' * 65}", flush=True)

    configs_a = []
    for V in class_configs:
        for seed in SEEDS:
            configs_a.append(dict(V=V, internal=64, seed=seed,
                                  max_attempts=16000, ticks=8))

    print(f"\n  Running {len(configs_a)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_a = list(pool.map(worker_baseline, configs_a))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    # Check for errors
    for r in results_a:
        if 'error' in r:
            print(f"  ERROR: V={r.get('V')} seed={r.get('seed')}: {r['error']}")

    # Table
    print(f"\n  {'V':>5s} {'N':>5s} {'Acc':>7s} {'Conns':>8s} {'Density':>8s} "
          f"{'Accept%':>8s} {'ms/att':>8s} {'Time':>7s} {'MaxConns':>10s}")
    print(f"  {'-' * 75}")

    phase_a_data = {}
    for V in class_configs:
        N = V + 64
        runs = [r for r in results_a if r.get('V') == V and 'error' not in r]
        if not runs:
            continue
        accs = [r['final_acc'] for r in runs]
        conns = [r['conns'] for r in runs]
        dens = [r['density'] for r in runs]
        ars = [r['accept_rate'] for r in runs]
        tpa = [r['time_per_att_ms'] for r in runs]
        tms = [r['time'] for r in runs]
        max_conns = N * (N - 1)

        phase_a_data[V] = {
            'acc': np.mean(accs), 'conns': np.mean(conns),
            'density': np.mean(dens), 'time_per_att': np.mean(tpa),
            'time': np.mean(tms),
        }

        print(f"  {V:5d} {N:5d} {np.mean(accs)*100:5.1f}% {np.mean(conns):7.0f} "
              f"{np.mean(dens):6.1f}% {np.mean(ars):6.2f}% "
              f"{np.mean(tpa):6.2f}ms {np.mean(tms):5.0f}s {max_conns:10d}")

    # Scaling curve summary
    print(f"\n  --- SCALING CURVE ---")
    print(f"  V -> Accuracy trend:")
    for V in class_configs:
        d = phase_a_data.get(V)
        if d:
            bar = '#' * int(d['acc'] * 50)
            print(f"  V={V:3d} (N={V+64:3d}): {d['acc']*100:5.1f}% {bar}")

    print(f"\n  V -> Cost trend (ms per attempt):")
    for V in class_configs:
        d = phase_a_data.get(V)
        if d:
            bar = '#' * min(int(d['time_per_att'] / 2), 50)
            print(f"  V={V:3d} (N={V+64:3d}): {d['time_per_att']:6.2f}ms {bar}")

    # ==========================================================
    # PHASE B: Neuron Scaling (V=64 fixed, internal grows)
    # ==========================================================
    internal_configs = [32, 64, 128, 256, 512]

    print(f"\n{'#' * 65}")
    print(f"  PHASE B: Neuron Scaling (V=64 fix, internal grows)")
    print(f"  internal={internal_configs}, N=64+internal, 16K att, 3 seeds")
    print(f"  {len(internal_configs) * len(SEEDS)} jobs")
    print(f"{'#' * 65}", flush=True)

    configs_b = []
    for internal in internal_configs:
        for seed in SEEDS:
            configs_b.append(dict(V=64, internal=internal, seed=seed,
                                  max_attempts=16000, ticks=8))

    print(f"\n  Running {len(configs_b)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_b = list(pool.map(worker_baseline, configs_b))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    for r in results_b:
        if 'error' in r:
            print(f"  ERROR: int={r.get('internal')} seed={r.get('seed')}: {r['error']}")

    print(f"\n  {'Internal':>8s} {'N':>5s} {'Acc':>7s} {'Conns':>8s} {'Density':>8s} "
          f"{'Accept%':>8s} {'ms/att':>8s} {'Time':>7s} {'MaxConns':>10s}")
    print(f"  {'-' * 80}")

    phase_b_data = {}
    for internal in internal_configs:
        N = 64 + internal
        runs = [r for r in results_b if r.get('internal') == internal and 'error' not in r]
        if not runs:
            continue
        accs = [r['final_acc'] for r in runs]
        conns = [r['conns'] for r in runs]
        dens = [r['density'] for r in runs]
        ars = [r['accept_rate'] for r in runs]
        tpa = [r['time_per_att_ms'] for r in runs]
        tms = [r['time'] for r in runs]
        max_conns = N * (N - 1)

        phase_b_data[internal] = {
            'acc': np.mean(accs), 'conns': np.mean(conns),
            'density': np.mean(dens), 'time_per_att': np.mean(tpa),
            'time': np.mean(tms), 'N': N,
        }

        print(f"  {internal:8d} {N:5d} {np.mean(accs)*100:5.1f}% {np.mean(conns):7.0f} "
              f"{np.mean(dens):6.1f}% {np.mean(ars):6.2f}% "
              f"{np.mean(tpa):6.2f}ms {np.mean(tms):5.0f}s {max_conns:10d}")

    # Scaling curve
    print(f"\n  --- NEURON SCALING CURVE ---")
    print(f"  Internal -> Accuracy:")
    for internal in internal_configs:
        d = phase_b_data.get(internal)
        if d:
            bar = '#' * int(d['acc'] * 50)
            print(f"  int={internal:3d} (N={d['N']:3d}): {d['acc']*100:5.1f}% {bar}")

    print(f"\n  Internal -> Cost (ms/att):")
    for internal in internal_configs:
        d = phase_b_data.get(internal)
        if d:
            bar = '#' * min(int(d['time_per_att'] / 5), 50)
            print(f"  int={internal:3d} (N={d['N']:3d}): {d['time_per_att']:7.2f}ms {bar}")

    print(f"\n  Internal -> Connections (actual vs max):")
    for internal in internal_configs:
        d = phase_b_data.get(internal)
        if d:
            N = d['N']
            max_c = N * (N - 1)
            print(f"  int={internal:3d} (N={N:3d}): {d['conns']:7.0f} / {max_c:7d} "
                  f"({d['density']:.1f}%)")

    # Connection scaling: N^2 linear or sublinear?
    if len(phase_b_data) >= 2:
        internals_sorted = sorted(phase_b_data.keys())
        print(f"\n  --- CONNECTION SCALING ANALYSIS ---")
        prev = None
        for internal in internals_sorted:
            d = phase_b_data[internal]
            if prev is not None:
                n_ratio = d['N'] / prev['N']
                conn_ratio = d['conns'] / max(prev['conns'], 1)
                time_ratio = d['time_per_att'] / max(prev['time_per_att'], 0.001)
                print(f"  N: {prev['N']:3d}->{d['N']:3d} ({n_ratio:.2f}x) | "
                      f"Conns: {conn_ratio:.2f}x | Time/att: {time_ratio:.2f}x")
            prev = d

    # ==========================================================
    # PHASE C: Scaling + Temperature
    # ==========================================================
    # Use the largest Phase B size that completed
    largest_internal = max(phase_b_data.keys()) if phase_b_data else 256
    largest_N = 64 + largest_internal

    print(f"\n{'#' * 65}")
    print(f"  PHASE C: Scaling + Temperature (V=64, N={largest_N})")
    print(f"  4 configs x 3 seeds = 12 jobs")
    print(f"  1. baseline NO CAP")
    print(f"  2. baseline + 30% density cap")
    print(f"  3. baseline + 15% density cap")
    print(f"  4. temp_gradual + 15% density cap")
    print(f"{'#' * 65}", flush=True)

    # Config 1: baseline no cap
    configs_c1 = [dict(V=64, internal=largest_internal, seed=s,
                       max_attempts=16000, ticks=8) for s in SEEDS]
    # Config 2: baseline + 30% cap
    configs_c2 = [dict(V=64, internal=largest_internal, seed=s,
                       max_density=0.30, max_attempts=16000, ticks=8)
                  for s in SEEDS]
    # Config 3: baseline + 15% cap
    configs_c3 = [dict(V=64, internal=largest_internal, seed=s,
                       max_density=0.15, max_attempts=16000, ticks=8)
                  for s in SEEDS]
    # Config 4: temp_gradual + 15% cap
    configs_c4 = [dict(V=64, internal=largest_internal, seed=s,
                       max_density=0.15, max_attempts=16000, ticks=8)
                  for s in SEEDS]

    all_c = ([(worker_baseline, c) for c in configs_c1] +
             [(worker_baseline_capped, c) for c in configs_c2] +
             [(worker_baseline_capped, c) for c in configs_c3] +
             [(worker_temp_capped, c) for c in configs_c4])

    def run_mixed(args):
        fn, cfg = args
        return fn(cfg)

    print(f"\n  Running 12 jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_c = list(pool.map(run_mixed, all_c))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    # Split results back
    r_c1 = results_c[0:3]
    r_c2 = results_c[3:6]
    r_c3 = results_c[6:9]
    r_c4 = results_c[9:12]

    configs_labels = [
        ("baseline NO CAP", r_c1),
        ("baseline +30% cap", r_c2),
        ("baseline +15% cap", r_c3),
        ("temp_gradual +15% cap", r_c4),
    ]

    print(f"\n  N={largest_N}, V=64, internal={largest_internal}")
    print(f"\n  {'Config':<25s} {'Acc':>7s} {'Conns':>8s} {'Density':>8s} "
          f"{'Accept%':>8s} {'ms/att':>8s} {'Time':>7s}")
    print(f"  {'-' * 80}")

    for label, runs in configs_labels:
        valid = [r for r in runs if 'error' not in r]
        if not valid:
            print(f"  {label:<25s} ERROR")
            for r in runs:
                if 'error' in r:
                    print(f"    {r['error']}")
            continue
        accs = [r['final_acc'] for r in valid]
        conns = [r['conns'] for r in valid]
        dens = [r['density'] for r in valid]
        ars = [r['accept_rate'] for r in valid]
        tpa = [r['time_per_att_ms'] for r in valid]
        tms = [r['time'] for r in valid]
        print(f"  {label:<25s} {np.mean(accs)*100:5.1f}% {np.mean(conns):7.0f} "
              f"{np.mean(dens):6.1f}% {np.mean(ars):6.2f}% "
              f"{np.mean(tpa):6.2f}ms {np.mean(tms):5.0f}s")

    # ==========================================================
    # FINAL VERDICT
    # ==========================================================
    print(f"\n  {'=' * 65}")
    print(f"  FINAL VERDICT")
    print(f"  {'=' * 65}")

    print(f"\n  Phase A -- Class Scaling (where does capacity run out?):")
    for V in class_configs:
        d = phase_a_data.get(V)
        if d:
            status = "OK" if d['acc'] > 0.9 else "HARD" if d['acc'] > 0.5 else "WALL"
            print(f"    V={V:3d}: {d['acc']*100:5.1f}% [{status}]")

    print(f"\n  Phase B -- Neuron Scaling (does more capacity help 64-class?):")
    for internal in internal_configs:
        d = phase_b_data.get(internal)
        if d:
            print(f"    int={internal:3d} (N={d['N']:3d}): {d['acc']*100:5.1f}% "
                  f"@ {d['time_per_att']:.1f}ms/att")

    print(f"\n  Phase C -- Brute force vs sparse @ N={largest_N}:")
    for label, runs in configs_labels:
        valid = [r for r in runs if 'error' not in r]
        if valid:
            acc = np.mean([r['final_acc'] for r in valid])
            conns = np.mean([r['conns'] for r in valid])
            print(f"    {label:<25s}: {acc*100:5.1f}% ({conns:.0f} conns)")

    # Key question answers
    print(f"\n  --- KEY QUESTIONS ---")

    # Q1: Where does class scaling break?
    wall_v = None
    for V in class_configs:
        d = phase_a_data.get(V)
        if d and d['acc'] < 0.5:
            wall_v = V
            break
    if wall_v:
        print(f"  Class wall: V={wall_v} (accuracy drops below 50%)")
    else:
        print(f"  Class wall: not reached (all V up to {class_configs[-1]} above 50%)")

    # Q2: Does more neurons help?
    if phase_b_data:
        best_int = max(phase_b_data.keys(), key=lambda k: phase_b_data[k]['acc'])
        worst_int = min(phase_b_data.keys(), key=lambda k: phase_b_data[k]['acc'])
        spread = phase_b_data[best_int]['acc'] - phase_b_data[worst_int]['acc']
        print(f"  Neuron scaling: best int={best_int} ({phase_b_data[best_int]['acc']*100:.1f}%), "
              f"worst int={worst_int} ({phase_b_data[worst_int]['acc']*100:.1f}%), "
              f"spread={spread*100:.1f}%")

    # Q3: Brute force vs temperature crossover?
    bf_acc = np.mean([r['final_acc'] for r in r_c1 if 'error' not in r]) if r_c1 else 0
    temp_acc = np.mean([r['final_acc'] for r in r_c4 if 'error' not in r]) if r_c4 else 0
    if temp_acc > bf_acc:
        print(f"  CROSSOVER FOUND: temp_gradual ({temp_acc*100:.1f}%) > "
              f"baseline ({bf_acc*100:.1f}%) at N={largest_N}")
    else:
        print(f"  NO CROSSOVER: baseline ({bf_acc*100:.1f}%) still > "
              f"temp_gradual ({temp_acc*100:.1f}%) at N={largest_N}")

    print(f"\n  {'=' * 65}")
    print(f"  DONE")
    print(f"  {'=' * 65}", flush=True)
