"""
INVERSE TICK TEST — RESUME from baseline
=========================================
Baseline already saved (47.5%). Runs remaining 5 conditions ONE AT A TIME
to avoid OOM. Each result saved immediately after completion.
"""

import numpy as np
import random
import time
import traceback
import json
import os
import gc
import sys
from datetime import datetime
from multiprocessing import Pool

# ============================================================
# Constants — SAME as original
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]
V = 64
INTERNAL = 128  # N = 64 + 128 + 64 = 192
N = V + INTERNAL + V
OUT_START = N - V
MAX_ATT = 64000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'inverse_tick_results.json')
PROGRESS_FILE = os.path.join(SCRIPT_DIR, 'inverse_tick_progress.txt')

# Use only 3 workers to reduce memory
N_WORKERS = 3


# ============================================================
# Infrastructure
# ============================================================

def log_progress(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    with open(PROGRESS_FILE, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ============================================================
# Network init
# ============================================================

def init_network(seed):
    np.random.seed(seed)
    random.seed(seed)
    r = np.random.rand(N, N)
    mask = np.zeros((N, N), dtype=np.float32)
    mask[r < 0.03] = -1
    mask[r > 0.97] = 1
    np.fill_diagonal(mask, 0)
    W = np.where(np.random.rand(N, N) > 0.5, np.float32(0.5), np.float32(1.5))
    perm = np.random.permutation(V)
    return mask, W, perm


# ============================================================
# Eval
# ============================================================

def eval_net(mask, W, perm, ticks, threshold=0.5, leak=0.85):
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
            probs = softmax(charge[OUT_START:OUT_START + V])
            if p == 1:
                if np.argmax(probs) == perm[i]:
                    correct += 1
                target_prob += probs[perm[i]]
    acc = correct / V
    tp = target_prob / V
    return acc, 0.5 * acc + 0.5 * tp


# ============================================================
# Mutations
# ============================================================

def add_connection(mask, W):
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


def rewire_connection(mask, W):
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


# ============================================================
# 2D Mood mutation
# ============================================================

def mutate_2d(mask, W, mood_x, mood_z):
    n_changes = max(1, int(1 + mood_z * 14))
    for _ in range(n_changes):
        if mood_x < 0.33:
            if random.random() < 0.7:
                add_connection(mask, W)
            else:
                flip_connection(mask)
        elif mood_x < 0.66:
            r = random.random()
            if r < 0.6:
                rewire_connection(mask, W)
            elif r < 0.8:
                flip_connection(mask)
            else:
                add_connection(mask, W)
        else:
            if random.random() < 0.5:
                flip_connection(mask)
            else:
                toggle_weight(mask, W)


# ============================================================
# Tick formulas
# ============================================================

def tick_inverse_v1(mood_x, mood_z):
    return max(1, min(12, int(2 + mood_x * 8 - mood_z * 3)))

def tick_inverse_v2(mood_x, mood_z):
    return max(2, min(12, int(1 + mood_x * 10 - mood_z * 4)))

def tick_inverse_v3(mood_x, mood_z):
    return max(2, min(12, int(4 + mood_x * 6 - mood_z * 2)))


# ============================================================
# Mood analysis
# ============================================================

def analyze_mood(history, floor_ceil_hits):
    if len(history) < 3:
        return {'floor_ceil': floor_ceil_hits, 'insufficient_data': True}
    mxs = [h['mx'] for h in history]
    mzs = [h['mz'] for h in history]
    ticks = [h['tick'] for h in history]
    n_chs = [h['n_ch'] for h in history]
    n = len(history)
    third = max(1, n // 3)
    first_mx, last_mx = np.mean(mxs[:third]), np.mean(mxs[-third:])
    first_mz, last_mz = np.mean(mzs[:third]), np.mean(mzs[-third:])
    first_tick, last_tick = np.mean(ticks[:third]), np.mean(ticks[-third:])
    scout_pct = sum(1 for m in mxs if m < 0.33) / n
    rewirer_pct = sum(1 for m in mxs if 0.33 <= m < 0.66) / n
    refiner_pct = sum(1 for m in mxs if m >= 0.66) / n
    tick_hist = {}
    for t in ticks:
        tick_hist[t] = tick_hist.get(t, 0) + 1
    return {
        'mood_x': {'start': round(first_mx, 4), 'end': round(last_mx, 4),
                    'mean': round(np.mean(mxs), 4), 'std': round(np.std(mxs), 4)},
        'mood_z': {'start': round(first_mz, 4), 'end': round(last_mz, 4),
                    'mean': round(np.mean(mzs), 4), 'std': round(np.std(mzs), 4)},
        'tick': {'start': round(first_tick, 2), 'end': round(last_tick, 2),
                 'mean': round(np.mean(ticks), 2), 'std': round(np.std(ticks), 2)},
        'phase_shift': {
            'x_delta': round(last_mx - first_mx, 4),
            'z_delta': round(last_mz - first_mz, 4),
            'tick_delta': round(last_tick - first_tick, 2),
        },
        'specialist_pct': {'scout': round(scout_pct, 4), 'rewirer': round(rewirer_pct, 4),
                           'refiner': round(refiner_pct, 4)},
        'tick_mean': round(np.mean(ticks), 2),
        'tick_hist': {str(k): v for k, v in sorted(tick_hist.items())},
        'nch_mean': round(np.mean(n_chs), 2),
        'floor_ceil': floor_ceil_hits,
    }


# ============================================================
# Training functions
# ============================================================

def train_baseline(seed):
    np.random.seed(seed)
    random.seed(seed)
    mask, W, perm = init_network(seed)
    acc, score = eval_net(mask, W, perm, ticks=8)
    best_acc = acc
    stale = 0
    for att in range(MAX_ATT):
        saved_mask = mask.copy()
        saved_W = W.copy()
        if stale < MAX_ATT // 3:
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
            if random.random() < 0.5:
                toggle_weight(mask, W)
            else:
                flip_connection(mask)
        new_acc, new_score = eval_net(mask, W, perm, ticks=8)
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
    return best_acc


def train_2d_mood_fix(seed, eval_tick):
    np.random.seed(seed)
    random.seed(seed)
    mask, W, perm = init_network(seed)
    acc, score = eval_net(mask, W, perm, ticks=eval_tick)
    best_acc = acc
    mood_x, mood_z = 0.5, 0.5
    mood_history = []
    floor_ceil_hits = {'x_floor': 0, 'x_ceil': 0, 'z_floor': 0, 'z_ceil': 0}
    for att in range(MAX_ATT):
        saved_mask = mask.copy()
        saved_W = W.copy()
        saved_mx, saved_mz = mood_x, mood_z
        if random.random() < 0.2:
            mood_x = float(np.clip(mood_x + random.gauss(0, 0.15), 0.0, 1.0))
        if random.random() < 0.2:
            mood_z = float(np.clip(mood_z + random.gauss(0, 0.15), 0.0, 1.0))
        if mood_x <= 0.01: floor_ceil_hits['x_floor'] += 1
        if mood_x >= 0.99: floor_ceil_hits['x_ceil'] += 1
        if mood_z <= 0.01: floor_ceil_hits['z_floor'] += 1
        if mood_z >= 0.99: floor_ceil_hits['z_ceil'] += 1
        mutate_2d(mask, W, mood_x, mood_z)
        new_acc, new_score = eval_net(mask, W, perm, ticks=eval_tick)
        if new_score > score:
            score = new_score
            best_acc = max(best_acc, new_acc)
            mood_history.append({
                'att': att, 'mx': round(mood_x, 4), 'mz': round(mood_z, 4),
                'tick': eval_tick, 'n_ch': max(1, int(1 + mood_z * 14)),
                'acc': round(new_acc, 4)
            })
        else:
            mask[:] = saved_mask
            W[:] = saved_W
            mood_x, mood_z = saved_mx, saved_mz
        if best_acc >= 0.99:
            break
    stats = analyze_mood(mood_history, floor_ceil_hits)
    stats['best_acc'] = best_acc
    stats['n_improvements'] = len(mood_history)
    stats['eval_tick'] = eval_tick
    return best_acc, stats


def train_inverse_tick(seed, tick_fn, label):
    np.random.seed(seed)
    random.seed(seed)
    mask, W, perm = init_network(seed)
    acc, score = eval_net(mask, W, perm, ticks=8)
    best_acc = acc
    mood_x, mood_z = 0.5, 0.5
    mood_history = []
    floor_ceil_hits = {'x_floor': 0, 'x_ceil': 0, 'z_floor': 0, 'z_ceil': 0}
    for att in range(MAX_ATT):
        saved_mask = mask.copy()
        saved_W = W.copy()
        saved_mx, saved_mz = mood_x, mood_z
        if random.random() < 0.2:
            mood_x = float(np.clip(mood_x + random.gauss(0, 0.15), 0.0, 1.0))
        if random.random() < 0.2:
            mood_z = float(np.clip(mood_z + random.gauss(0, 0.15), 0.0, 1.0))
        if mood_x <= 0.01: floor_ceil_hits['x_floor'] += 1
        if mood_x >= 0.99: floor_ceil_hits['x_ceil'] += 1
        if mood_z <= 0.01: floor_ceil_hits['z_floor'] += 1
        if mood_z >= 0.99: floor_ceil_hits['z_ceil'] += 1
        tick = tick_fn(mood_x, mood_z)
        mutate_2d(mask, W, mood_x, mood_z)
        new_acc, new_score = eval_net(mask, W, perm, ticks=tick)
        if new_score > score:
            score = new_score
            best_acc = max(best_acc, new_acc)
            mood_history.append({
                'att': att, 'mx': round(mood_x, 4), 'mz': round(mood_z, 4),
                'tick': tick, 'n_ch': max(1, int(1 + mood_z * 14)),
                'acc': round(new_acc, 4)
            })
        else:
            mask[:] = saved_mask
            W[:] = saved_W
            mood_x, mood_z = saved_mx, saved_mz
        if best_acc >= 0.99:
            break
    stats = analyze_mood(mood_history, floor_ceil_hits)
    stats['best_acc'] = best_acc
    stats['n_improvements'] = len(mood_history)
    stats['label'] = label
    return best_acc, stats


# ============================================================
# Worker wrappers
# ============================================================

def worker_fix8(seed):
    try:
        best_acc, stats = train_2d_mood_fix(seed, eval_tick=8)
        return {'seed': seed, 'acc': best_acc, 'stats': stats}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}

def worker_fix4(seed):
    try:
        best_acc, stats = train_2d_mood_fix(seed, eval_tick=4)
        return {'seed': seed, 'acc': best_acc, 'stats': stats}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}

def worker_inverse_v1(seed):
    try:
        best_acc, stats = train_inverse_tick(seed, tick_inverse_v1, 'inverse_v1')
        return {'seed': seed, 'acc': best_acc, 'stats': stats}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}

def worker_inverse_v2(seed):
    try:
        best_acc, stats = train_inverse_tick(seed, tick_inverse_v2, 'inverse_v2')
        return {'seed': seed, 'acc': best_acc, 'stats': stats}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}

def worker_inverse_v3(seed):
    try:
        best_acc, stats = train_inverse_tick(seed, tick_inverse_v3, 'inverse_v3')
        return {'seed': seed, 'acc': best_acc, 'stats': stats}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}


# ============================================================
# Run one condition safely
# ============================================================

def run_one(label, worker_fn, step_num, total, results):
    """Run a single condition. Save immediately. Force GC after."""
    if label in results:
        log_progress(f"  [{step_num}/{total}] SKIP {label} — already done (avg={results[label]['avg_acc']:.1%})")
        return

    log_progress(f"\n  [{step_num}/{total}] START: {label}")
    t0 = time.time()

    # Run with pool — limited workers
    with Pool(N_WORKERS) as pool:
        res_list = pool.map(worker_fn, SEEDS)

    elapsed = time.time() - t0
    accs = [r['acc'] for r in res_list]

    for r in res_list:
        err = f" ERROR: {r['error']}" if 'error' in r else ""
        log_progress(f"    seed={r['seed']}: acc={r['acc']:.1%}{err}")
    log_progress(f"  {label}: avg={np.mean(accs):.1%} best={max(accs):.1%} time={elapsed:.1f}s")

    # Mood stats
    for r in res_list:
        s = r.get('stats', {})
        if 'mood_x' not in s:
            continue
        spec = s.get('specialist_pct', {})
        tick_info = s.get('tick', {})
        log_progress(f"    seed={r['seed']} mood: "
                     f"x={s['mood_x']['mean']:.2f}({s['mood_x']['start']:.2f}->{s['mood_x']['end']:.2f}) "
                     f"z={s['mood_z']['mean']:.2f}({s['mood_z']['start']:.2f}->{s['mood_z']['end']:.2f}) "
                     f"tick_avg={s.get('tick_mean', '?')} "
                     f"tick({tick_info.get('start', '?')}->{tick_info.get('end', '?')}) "
                     f"scout={spec.get('scout', 0):.0%} rewirer={spec.get('rewirer', 0):.0%} "
                     f"refiner={spec.get('refiner', 0):.0%} "
                     f"nch={s.get('nch_mean', '?')} "
                     f"impr={s.get('n_improvements', '?')}")
        log_progress(f"             floor/ceil: {s.get('floor_ceil', {})}")

    results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'details': res_list
    }
    save_results(results)
    log_progress(f"  >>> SAVED to {RESULTS_FILE}")

    # Force garbage collection to free memory
    gc.collect()
    time.sleep(2)


# ============================================================
# MAIN — resume-safe
# ============================================================

def main():
    results = load_results()

    log_progress("\n" + "=" * 70)
    log_progress("INVERSE TICK TEST — RESUME RUN")
    log_progress(f"V={V}, N={N}, attempts={MAX_ATT}, seeds={SEEDS}, workers={N_WORKERS}")
    log_progress(f"Already completed: {list(results.keys())}")
    log_progress("=" * 70)

    conditions = [
        ('baseline_V64_64K', None),  # skip if exists
        ('2d_mood_fix8_V64_64K', worker_fix8),
        ('2d_mood_fix4_V64_64K', worker_fix4),
        ('inverse_v1_V64_64K', worker_inverse_v1),
        ('inverse_v2_V64_64K', worker_inverse_v2),
        ('inverse_v3_V64_64K', worker_inverse_v3),
    ]

    total = len(conditions)
    for i, (label, worker_fn) in enumerate(conditions, 1):
        if label in results:
            log_progress(f"  [{i}/{total}] SKIP {label} — already done (avg={results[label]['avg_acc']:.1%})")
            continue
        if worker_fn is None:
            log_progress(f"  [{i}/{total}] SKIP {label} — no worker (baseline needs original script)")
            continue
        run_one(label, worker_fn, i, total, results)

    # FINAL SUMMARY
    log_progress("\n" + "=" * 70)
    log_progress("FINAL SUMMARY")
    log_progress("=" * 70)
    log_progress(f"{'Method':<30} {'Avg Acc':>10} {'Best Acc':>10} {'Time':>10}")
    log_progress("-" * 70)

    order = ['baseline_V64_64K', '2d_mood_fix8_V64_64K', '2d_mood_fix4_V64_64K',
             'inverse_v1_V64_64K', 'inverse_v2_V64_64K', 'inverse_v3_V64_64K']
    for k in order:
        if k in results:
            r = results[k]
            log_progress(f"{k:<30} {r['avg_acc']:>9.1%} {r['best_acc']:>9.1%} {r['time']:>9.1f}s")

    # Verdict
    fix8_avg = results.get('2d_mood_fix8_V64_64K', {}).get('avg_acc', 0)
    for inv_label in ['inverse_v1_V64_64K', 'inverse_v2_V64_64K', 'inverse_v3_V64_64K']:
        if inv_label in results:
            inv_avg = results[inv_label]['avg_acc']
            diff = inv_avg - fix8_avg
            if diff > 0.02:
                verdict = f"INVERSE WINS by {diff:+.1%}"
            elif diff > -0.02:
                verdict = f"DRAW ({diff:+.1%})"
            else:
                verdict = f"FIX8 WINS by {-diff:.1%}"
            log_progress(f"  {inv_label} vs fix8: {diff:+.1%} → {verdict}")

    log_progress("\nDONE.")


if __name__ == '__main__':
    main()
