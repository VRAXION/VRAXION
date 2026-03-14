"""
LEARNABLE MOOD 3D TEST — The network learns its OWN search strategy
====================================================================
3 extra float params (mood_x, mood_y, mood_z) co-evolve with the network.
  mood_x: WHAT to do    (0=scout, 0.5=rewirer, 1=refiner)
  mood_y: HOW LONG       (0=1tick, 1=12ticks)
  mood_z: HOW MUCH       (0=1 conn, 1=15 conns per step)

Phase 1: V=16  (validation)    — 8K attempts
Phase 2: V=32                  — 16K attempts
Phase 3: V=64                  — 16K attempts
Phase 4: V=64 LONG RUN         — 64K attempts

Each phase: 3d_mood vs random vs baseline, 5 seeds, parallel.
"""

import numpy as np
import math
import random
import time
import traceback
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ============================================================
# Constants
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]
INTERNAL = 64

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'mood3d_results.json')
PROGRESS_FILE = os.path.join(SCRIPT_DIR, 'mood3d_progress.txt')

results = {}
N_WORKERS = min(cpu_count(), len(SEEDS))


# ============================================================
# Infrastructure
# ============================================================

def log_progress(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    with open(PROGRESS_FILE, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def save_results():
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ============================================================
# SplitIONet (no highway, clean copy for this test)
# ============================================================

class SplitIONet:
    def __init__(self, vocab, internal=64, density=0.06, threshold=0.5, leak=0.85):
        self.V = vocab
        self.internal = internal
        self.threshold = threshold
        self.leak = leak
        self.N = vocab + internal + vocab
        self.out_start = self.N - vocab
        N = self.N

        r = np.random.rand(N, N)
        self.mask = np.zeros((N, N), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        self.W = np.where(
            np.random.rand(N, N) > 0.5,
            np.float32(0.5), np.float32(1.5))

        self.state = np.zeros(N, dtype=np.float32)
        self.charge = np.zeros(N, dtype=np.float32)

    def reset(self):
        self.state *= 0
        self.charge *= 0

    def forward(self, world, ticks=6):
        act = self.state.copy()
        Weff = self.W * self.mask
        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = act @ Weff + act * 0.1
            self.charge += raw * 0.3
            self.charge *= self.leak
            act = np.maximum(self.charge - self.threshold, 0.0)
            self.charge = np.clip(self.charge, -self.threshold * 2, self.threshold * 2)
        self.state = act.copy()
        return self.charge[self.out_start:]

    def count_connections(self):
        return int((self.mask != 0).sum())


# ============================================================
# Scoring
# ============================================================

def eval_net(mask, W, perm, V, N, out_start, ticks, threshold=0.5, leak=0.85):
    """Standalone eval without class overhead — 2-pass, combined scoring."""
    state = np.zeros(N, dtype=np.float32)
    charge = np.zeros(N, dtype=np.float32)
    Weff = W * mask
    correct = 0
    target_prob = 0.0
    for p in range(2):
        for i in range(V):
            world = np.zeros(V, dtype=np.float32)
            world[i] = 1.0
            act = state.copy()
            ch = charge.copy()
            for t in range(ticks):
                if t == 0:
                    act[:V] = world
                raw = act @ Weff + act * 0.1
                ch += raw * 0.3
                ch *= leak
                act = np.maximum(ch - threshold, 0.0)
                ch = np.clip(ch, -threshold * 2, threshold * 2)
            state = act.copy()
            charge = ch.copy()
            probs = softmax(charge[out_start:out_start + V])
            if p == 1:
                if np.argmax(probs) == perm[i]:
                    correct += 1
                target_prob += probs[perm[i]]
    acc = correct / V
    tp = target_prob / V
    return acc, 0.5 * acc + 0.5 * tp


# ============================================================
# 3D Mood Mutation
# ============================================================

def add_connection(mask, W, N):
    dead = np.argwhere(mask == 0)
    dead = dead[dead[:, 0] != dead[:, 1]]
    if len(dead) > 0:
        i = dead[random.randint(0, len(dead) - 1)]
        mask[i[0], i[1]] = 1.0 if random.random() > 0.5 else -1.0
        W[i[0], i[1]] = random.choice([np.float32(0.5), np.float32(1.5)])


def flip_connection(mask):
    alive = np.argwhere(mask != 0)
    if len(alive) > 0:
        i = alive[random.randint(0, len(alive) - 1)]
        mask[i[0], i[1]] *= -1


def rewire_connection(mask, W, N):
    alive = np.argwhere(mask != 0)
    if len(alive) > 0:
        i = alive[random.randint(0, len(alive) - 1)]
        old_sign = mask[i[0], i[1]]
        old_w = W[i[0], i[1]]
        mask[i[0], i[1]] = 0
        nc = random.randint(0, N - 1)
        while nc == i[0]:
            nc = random.randint(0, N - 1)
        mask[i[0], nc] = old_sign
        W[i[0], nc] = old_w


def toggle_weight(mask, W):
    alive = np.argwhere(mask != 0)
    if len(alive) > 0:
        i = alive[random.randint(0, len(alive) - 1)]
        W[i[0], i[1]] = np.float32(1.5) if W[i[0], i[1]] < 1.0 else np.float32(0.5)


def mutate_3d(mask, W, mood_x, mood_y, mood_z, N):
    """3 mood floats determine mutation TYPE, DEPTH, INTENSITY."""
    ticks = max(1, int(1 + mood_y * 11))  # 1..12
    n_changes = max(1, int(1 + mood_z * 14))  # 1..15

    for _ in range(n_changes):
        if mood_x < 0.33:
            # SCOUT: add heavy (70% add, 30% flip)
            if random.random() < 0.7:
                add_connection(mask, W, N)
            else:
                flip_connection(mask)
        elif mood_x < 0.66:
            # REWIRER: rewire heavy (60% rewire, 20% flip, 20% add)
            r = random.random()
            if r < 0.6:
                rewire_connection(mask, W, N)
            elif r < 0.8:
                flip_connection(mask)
            else:
                add_connection(mask, W, N)
        else:
            # REFINER: flip + weight (50/50)
            if random.random() < 0.5:
                flip_connection(mask)
            else:
                toggle_weight(mask, W)

    return ticks


# ============================================================
# Training: 3D Mood
# ============================================================

def train_3d_mood(seed, V, internal, max_attempts=16000):
    """Train with learnable 3D mood. Returns (best_acc, mood_history, stats)."""
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal + V
    out_start = N - V
    perm = np.random.permutation(V)

    # Init network
    r = np.random.rand(N, N)
    mask = np.zeros((N, N), dtype=np.float32)
    mask[r < 0.03] = -1
    mask[r > 0.97] = 1
    np.fill_diagonal(mask, 0)
    W = np.where(np.random.rand(N, N) > 0.5, np.float32(0.5), np.float32(1.5))

    # Initial eval with default ticks
    acc, score = eval_net(mask, W, perm, V, N, out_start, ticks=8)
    best_acc = acc

    # Init mood — neutral
    mood_x, mood_y, mood_z = 0.5, 0.5, 0.5

    mood_history = []
    floor_ceil_hits = {'x_floor': 0, 'x_ceil': 0, 'y_floor': 0, 'y_ceil': 0,
                       'z_floor': 0, 'z_ceil': 0}

    for att in range(max_attempts):
        # Save state
        saved_mask = mask.copy()
        saved_W = W.copy()
        saved_mx, saved_my, saved_mz = mood_x, mood_y, mood_z

        # Mutate mood (20% chance per axis)
        if random.random() < 0.2:
            mood_x = float(np.clip(mood_x + random.gauss(0, 0.15), 0.0, 1.0))
        if random.random() < 0.2:
            mood_y = float(np.clip(mood_y + random.gauss(0, 0.15), 0.0, 1.0))
        if random.random() < 0.2:
            mood_z = float(np.clip(mood_z + random.gauss(0, 0.15), 0.0, 1.0))

        # Track floor/ceil
        if mood_x <= 0.01: floor_ceil_hits['x_floor'] += 1
        if mood_x >= 0.99: floor_ceil_hits['x_ceil'] += 1
        if mood_y <= 0.01: floor_ceil_hits['y_floor'] += 1
        if mood_y >= 0.99: floor_ceil_hits['y_ceil'] += 1
        if mood_z <= 0.01: floor_ceil_hits['z_floor'] += 1
        if mood_z >= 0.99: floor_ceil_hits['z_ceil'] += 1

        # Mutate network (guided by mood)
        ticks = mutate_3d(mask, W, mood_x, mood_y, mood_z, N)

        # Eval
        new_acc, new_score = eval_net(mask, W, perm, V, N, out_start, ticks=ticks)

        if new_score > score:
            score = new_score
            best_acc = max(best_acc, new_acc)
            mood_history.append({
                'att': att, 'mx': round(mood_x, 4), 'my': round(mood_y, 4),
                'mz': round(mood_z, 4),
                'tick': ticks, 'n_ch': max(1, int(1 + mood_z * 14)),
                'acc': round(new_acc, 4)
            })
        else:
            # Revert everything
            mask[:] = saved_mask
            W[:] = saved_W
            mood_x, mood_y, mood_z = saved_mx, saved_my, saved_mz

        if best_acc >= 0.99:
            break

    # Analyze mood history
    stats = analyze_mood(mood_history, max_attempts, floor_ceil_hits)
    stats['best_acc'] = best_acc
    stats['n_improvements'] = len(mood_history)
    stats['conns'] = int((mask != 0).sum())

    return best_acc, mood_history, stats


def analyze_mood(history, max_attempts, floor_ceil_hits):
    """Analyze mood convergence and phase shifts."""
    if len(history) < 3:
        return {'floor_ceil': floor_ceil_hits, 'insufficient_data': True}

    mxs = [h['mx'] for h in history]
    mys = [h['my'] for h in history]
    mzs = [h['mz'] for h in history]
    ticks = [h['tick'] for h in history]
    n_chs = [h['n_ch'] for h in history]

    n = len(history)
    third = max(1, n // 3)

    # Phase analysis: first third vs last third
    first_mx = np.mean(mxs[:third])
    last_mx = np.mean(mxs[-third:])
    first_my = np.mean(mys[:third])
    last_my = np.mean(mys[-third:])
    first_mz = np.mean(mzs[:third])
    last_mz = np.mean(mzs[-third:])

    # Specialist distribution
    scout_pct = sum(1 for m in mxs if m < 0.33) / n
    rewirer_pct = sum(1 for m in mxs if 0.33 <= m < 0.66) / n
    refiner_pct = sum(1 for m in mxs if m >= 0.66) / n

    # Tick distribution
    tick_hist = {}
    for t in ticks:
        tick_hist[t] = tick_hist.get(t, 0) + 1

    # N_changes distribution
    nch_hist = {}
    for c in n_chs:
        nch_hist[c] = nch_hist.get(c, 0) + 1

    return {
        'mood_x': {'start': round(first_mx, 4), 'end': round(last_mx, 4),
                    'mean': round(np.mean(mxs), 4), 'std': round(np.std(mxs), 4)},
        'mood_y': {'start': round(first_my, 4), 'end': round(last_my, 4),
                    'mean': round(np.mean(mys), 4), 'std': round(np.std(mys), 4)},
        'mood_z': {'start': round(first_mz, 4), 'end': round(last_mz, 4),
                    'mean': round(np.mean(mzs), 4), 'std': round(np.std(mzs), 4)},
        'phase_shift': {
            'x_delta': round(last_mx - first_mx, 4),
            'y_delta': round(last_my - first_my, 4),
            'z_delta': round(last_mz - first_mz, 4),
        },
        'specialist_pct': {'scout': round(scout_pct, 4), 'rewirer': round(rewirer_pct, 4),
                           'refiner': round(refiner_pct, 4)},
        'tick_mean': round(np.mean(ticks), 2),
        'tick_hist': {str(k): v for k, v in sorted(tick_hist.items())},
        'nch_mean': round(np.mean(n_chs), 2),
        'nch_hist': {str(k): v for k, v in sorted(nch_hist.items())},
        'floor_ceil': floor_ceil_hits,
    }


# ============================================================
# Training: Random mood (control)
# ============================================================

def train_random_mood(seed, V, internal, max_attempts=16000):
    """Random mood each step — no learning. Control group."""
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal + V
    out_start = N - V
    perm = np.random.permutation(V)

    r = np.random.rand(N, N)
    mask = np.zeros((N, N), dtype=np.float32)
    mask[r < 0.03] = -1
    mask[r > 0.97] = 1
    np.fill_diagonal(mask, 0)
    W = np.where(np.random.rand(N, N) > 0.5, np.float32(0.5), np.float32(1.5))

    acc, score = eval_net(mask, W, perm, V, N, out_start, ticks=8)
    best_acc = acc

    for att in range(max_attempts):
        saved_mask = mask.copy()
        saved_W = W.copy()

        # Random mood each step (NOT learned)
        mood_x = random.random()
        mood_y = random.random()
        mood_z = random.random()

        ticks = mutate_3d(mask, W, mood_x, mood_y, mood_z, N)
        new_acc, new_score = eval_net(mask, W, perm, V, N, out_start, ticks=ticks)

        if new_score > score:
            score = new_score
            best_acc = max(best_acc, new_acc)
        else:
            mask[:] = saved_mask
            W[:] = saved_W

        if best_acc >= 0.99:
            break

    return best_acc, int((mask != 0).sum())


# ============================================================
# Training: Baseline (refiner only, tick=8)
# ============================================================

def train_baseline(seed, V, internal, max_attempts=16000):
    """Classic refiner-only hill climbing. tick=8 fixed."""
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal + V
    out_start = N - V
    perm = np.random.permutation(V)

    r = np.random.rand(N, N)
    mask = np.zeros((N, N), dtype=np.float32)
    mask[r < 0.03] = -1
    mask[r > 0.97] = 1
    np.fill_diagonal(mask, 0)
    W = np.where(np.random.rand(N, N) > 0.5, np.float32(0.5), np.float32(1.5))

    acc, score = eval_net(mask, W, perm, V, N, out_start, ticks=8)
    best_acc = acc
    stale = 0

    for att in range(max_attempts):
        saved_mask = mask.copy()
        saved_W = W.copy()

        # Classic: structure or weight mutation
        if stale < max_attempts // 3:
            # Structure phase
            alive = np.argwhere(mask != 0)
            if len(alive) > 0 and random.random() < 0.3:
                j = alive[random.randint(0, len(alive) - 1)]
                mask[j[0], j[1]] *= -1
            else:
                dead = np.argwhere(mask == 0)
                dead = dead[dead[:, 0] != dead[:, 1]]
                if len(dead) > 0:
                    j = dead[random.randint(0, len(dead) - 1)]
                    mask[j[0], j[1]] = 1.0 if random.random() > 0.5 else -1.0
        else:
            # Mixed
            if random.random() < 0.5:
                toggle_weight(mask, W)
            else:
                flip_connection(mask)

        new_acc, new_score = eval_net(mask, W, perm, V, N, out_start, ticks=8)

        if new_score > score:
            score = new_score
            best_acc = max(best_acc, new_acc)
            stale = 0
        else:
            mask[:] = saved_mask
            W[:] = saved_W
            stale += 1

        if best_acc >= 0.99:
            break

    return best_acc, int((mask != 0).sum())


# ============================================================
# Parallel worker wrappers
# ============================================================

def worker_3d_mood(args):
    seed, V, internal, max_att = args
    try:
        best_acc, history, stats = train_3d_mood(seed, V, internal, max_att)
        return {'seed': seed, 'acc': best_acc, 'stats': stats,
                'history_len': len(history), 'history_sample': history[-5:] if history else []}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}


def worker_random(args):
    seed, V, internal, max_att = args
    try:
        best_acc, conns = train_random_mood(seed, V, internal, max_att)
        return {'seed': seed, 'acc': best_acc, 'conns': conns}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}


def worker_baseline(args):
    seed, V, internal, max_att = args
    try:
        best_acc, conns = train_baseline(seed, V, internal, max_att)
        return {'seed': seed, 'acc': best_acc, 'conns': conns}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}


# ============================================================
# Run a phase
# ============================================================

def run_phase(phase_name, V, max_att):
    log_progress("=" * 60)
    log_progress(f"PHASE: {phase_name} | V={V} | attempts={max_att}")
    log_progress("=" * 60)

    phase_results = {}

    # --- 3D MOOD ---
    label = f"3d_mood_V{V}_{max_att // 1000}K"
    log_progress(f"  START: {label}")
    t0 = time.time()
    args = [(s, V, INTERNAL, max_att) for s in SEEDS]
    with Pool(N_WORKERS) as pool:
        mood_results = pool.map(worker_3d_mood, args)
    elapsed = time.time() - t0

    accs = [r['acc'] for r in mood_results]
    for r in mood_results:
        log_progress(f"    seed={r['seed']}: acc={r['acc']:.1%}")
    log_progress(f"  {label}: avg={np.mean(accs):.1%} best={max(accs):.1%} time={elapsed:.1f}s")

    # Mood analysis summary
    for r in mood_results:
        if 'stats' in r and 'mood_x' in r.get('stats', {}):
            s = r['stats']
            spec = s.get('specialist_pct', {})
            log_progress(f"    seed={r['seed']} mood: "
                         f"x={s['mood_x']['mean']:.2f}({s['mood_x']['start']:.2f}->{s['mood_x']['end']:.2f}) "
                         f"y={s['mood_y']['mean']:.2f}({s['mood_y']['start']:.2f}->{s['mood_y']['end']:.2f}) "
                         f"z={s['mood_z']['mean']:.2f}({s['mood_z']['start']:.2f}->{s['mood_z']['end']:.2f}) "
                         f"scout={spec.get('scout', 0):.0%} rewirer={spec.get('rewirer', 0):.0%} "
                         f"refiner={spec.get('refiner', 0):.0%} tick_avg={s.get('tick_mean', '?')}")

    phase_results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'details': mood_results
    }
    save_phase(phase_results)
    time.sleep(2)

    # --- RANDOM MOOD (control) ---
    label = f"random_V{V}_{max_att // 1000}K"
    log_progress(f"  START: {label}")
    t0 = time.time()
    args = [(s, V, INTERNAL, max_att) for s in SEEDS]
    with Pool(N_WORKERS) as pool:
        rand_results = pool.map(worker_random, args)
    elapsed = time.time() - t0

    accs = [r['acc'] for r in rand_results]
    for r in rand_results:
        log_progress(f"    seed={r['seed']}: acc={r['acc']:.1%}")
    log_progress(f"  {label}: avg={np.mean(accs):.1%} best={max(accs):.1%} time={elapsed:.1f}s")

    phase_results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'details': rand_results
    }
    save_phase(phase_results)
    time.sleep(2)

    # --- BASELINE ---
    label = f"baseline_V{V}_{max_att // 1000}K"
    log_progress(f"  START: {label}")
    t0 = time.time()
    args = [(s, V, INTERNAL, max_att) for s in SEEDS]
    with Pool(N_WORKERS) as pool:
        base_results = pool.map(worker_baseline, args)
    elapsed = time.time() - t0

    accs = [r['acc'] for r in base_results]
    for r in base_results:
        log_progress(f"    seed={r['seed']}: acc={r['acc']:.1%}")
    log_progress(f"  {label}: avg={np.mean(accs):.1%} best={max(accs):.1%} time={elapsed:.1f}s")

    phase_results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'details': base_results
    }
    save_phase(phase_results)
    time.sleep(2)

    # Summary
    keys = list(phase_results.keys())
    log_progress(f"  PHASE SUMMARY:")
    for k in keys:
        log_progress(f"    {k}: avg={phase_results[k]['avg_acc']:.1%} best={phase_results[k]['best_acc']:.1%}")

    return phase_results


def save_phase(phase_results):
    """Merge phase results into global and save."""
    results.update(phase_results)
    save_results()


# ============================================================
# MAIN
# ============================================================

def main():
    log_progress("LEARNABLE MOOD 3D TEST STARTED")
    log_progress(f"Seeds: {SEEDS}, Internal: {INTERNAL}, Workers: {N_WORKERS}")

    # Phase 1: V=16, 8K attempts (validation)
    try:
        p1 = run_phase("Phase 1: V=16 VALIDATION", V=16, max_att=8000)
    except Exception as e:
        log_progress(f"PHASE 1 CRASH: {traceback.format_exc()}")
    time.sleep(2)

    # Phase 2: V=32, 16K attempts
    try:
        p2 = run_phase("Phase 2: V=32", V=32, max_att=16000)
    except Exception as e:
        log_progress(f"PHASE 2 CRASH: {traceback.format_exc()}")
    time.sleep(2)

    # Phase 3: V=64, 16K attempts
    try:
        p3 = run_phase("Phase 3: V=64", V=64, max_att=16000)
    except Exception as e:
        log_progress(f"PHASE 3 CRASH: {traceback.format_exc()}")
    time.sleep(2)

    # Phase 4: V=64, 64K attempts (LONG RUN — no random, just mood vs baseline)
    log_progress("=" * 60)
    log_progress("PHASE 4: V=64 LONG RUN | attempts=64K")
    log_progress("=" * 60)

    try:
        # 3D Mood 64K
        label = "3d_mood_V64_64K"
        log_progress(f"  START: {label}")
        t0 = time.time()
        args = [(s, 64, INTERNAL, 64000) for s in SEEDS]
        with Pool(N_WORKERS) as pool:
            mood_long = pool.map(worker_3d_mood, args)
        elapsed = time.time() - t0

        accs = [r['acc'] for r in mood_long]
        for r in mood_long:
            log_progress(f"    seed={r['seed']}: acc={r['acc']:.1%}")
        log_progress(f"  {label}: avg={np.mean(accs):.1%} best={max(accs):.1%} time={elapsed:.1f}s")

        for r in mood_long:
            if 'stats' in r and 'mood_x' in r.get('stats', {}):
                s = r['stats']
                spec = s.get('specialist_pct', {})
                ps = s.get('phase_shift', {})
                log_progress(f"    seed={r['seed']} JOURNEY: "
                             f"x_shift={ps.get('x_delta', '?'):+.3f} "
                             f"y_shift={ps.get('y_delta', '?'):+.3f} "
                             f"z_shift={ps.get('z_delta', '?'):+.3f} "
                             f"tick_avg={s.get('tick_mean', '?')} "
                             f"nch_avg={s.get('nch_mean', '?')} "
                             f"improvements={s.get('n_improvements', '?')}")
                log_progress(f"             floor/ceil: {s.get('floor_ceil', {})}")

        results[label] = {
            'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
            'time': elapsed, 'details': mood_long
        }
        save_results()
        time.sleep(2)

        # Baseline 64K
        label = "baseline_V64_64K"
        log_progress(f"  START: {label}")
        t0 = time.time()
        args = [(s, 64, INTERNAL, 64000) for s in SEEDS]
        with Pool(N_WORKERS) as pool:
            base_long = pool.map(worker_baseline, args)
        elapsed = time.time() - t0

        accs = [r['acc'] for r in base_long]
        for r in base_long:
            log_progress(f"    seed={r['seed']}: acc={r['acc']:.1%}")
        log_progress(f"  {label}: avg={np.mean(accs):.1%} best={max(accs):.1%} time={elapsed:.1f}s")

        results[label] = {
            'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
            'time': elapsed, 'details': base_long
        }
        save_results()

    except Exception as e:
        log_progress(f"PHASE 4 CRASH: {traceback.format_exc()}")

    log_progress("=" * 60)
    log_progress("ALL PHASES COMPLETE")
    log_progress("=" * 60)

    # Final summary
    log_progress("FINAL RESULTS:")
    for k, v in results.items():
        if isinstance(v, dict) and 'avg_acc' in v:
            log_progress(f"  {k}: avg={v['avg_acc']:.1%} best={v['best_acc']:.1%}")


if __name__ == '__main__':
    main()
