"""
2D MOOD FIX EVAL TEST — The tick=1 greedy shortcut fix
=======================================================
Problem: 3D mood learns tick=1 ("greedy shortcut") — fast eval = more attempts
short-term, but shallow eval = bad mutations kept = worse long-term.

Solution: REMOVE the tick knob. Fix eval at tick=8 (or tick=4).
Keep mood_x (WHAT) and mood_z (HOW MUCH) learnable.

MAIN TEST: V=64, 64K attempts, 5 seeds
  1. baseline      — refiner only, tick=8
  2. 3d_mood       — mood_x+y+z, tick learnable (CONTROL)
  3. 2d_mood_fix8  — mood_x+z, tick=8 FIX        ← THE QUESTION
  4. 2d_mood_fix4  — mood_x+z, tick=4 FIX        (faster eval)

SIDE QUEST 1: Byte vs One-Hot encoding (V=64, 8K, 5 seeds)
SIDE QUEST 2: Intrinsic Dimension Calibration (V=16/32/64, 8K, 5 seeds)
SIDE QUEST 3: Context length with byte_8 encoding (bigram, 8K, 3 seeds)
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
SEEDS_SHORT = [42, 123, 777]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'mood2d_results.json')
PROGRESS_FILE = os.path.join(SCRIPT_DIR, 'mood2d_progress.txt')

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
# Network init (shared by all methods)
# ============================================================

def init_network(V, N, seed, density=0.06):
    """Initialize mask, W, permutation with given seed."""
    np.random.seed(seed)
    random.seed(seed)

    r = np.random.rand(N, N)
    mask = np.zeros((N, N), dtype=np.float32)
    mask[r < density / 2] = -1
    mask[r > 1 - density / 2] = 1
    np.fill_diagonal(mask, 0)
    W = np.where(np.random.rand(N, N) > 0.5, np.float32(0.5), np.float32(1.5))

    perm = np.random.permutation(V)
    return mask, W, perm


# ============================================================
# Eval: one-hot encoding (standard)
# ============================================================

def eval_net(mask, W, perm, V, N, out_start, ticks, threshold=0.5, leak=0.85):
    """Standalone eval — 2-pass, combined scoring, one-hot encoding."""
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
# Eval: byte_8 encoding
# ============================================================

def encode_byte8(value, n_bits=8):
    """Encode an integer as 8-bit binary vector."""
    bits = np.zeros(n_bits, dtype=np.float32)
    for b in range(n_bits):
        if value & (1 << b):
            bits[b] = 1.0
    return bits


def eval_net_byte8(mask, W, perm, V, N, input_size, output_size, out_start,
                   ticks, threshold=0.5, leak=0.85):
    """Eval with byte_8 encoding — 8 bits per symbol for input AND output."""
    state = np.zeros(N, dtype=np.float32)
    charge = np.zeros(N, dtype=np.float32)
    Weff = W * mask
    correct = 0
    target_prob = 0.0
    n_bits = 8

    for p in range(2):
        for i in range(V):
            world = encode_byte8(i, n_bits)
            act = state.copy()
            ch = charge.copy()
            for t in range(ticks):
                if t == 0:
                    act[:input_size] = 0.0
                    act[:n_bits] = world
                raw = act @ Weff + act * 0.1
                ch += raw * 0.3
                ch *= leak
                act = np.maximum(ch - threshold, 0.0)
                ch = np.clip(ch, -threshold * 2, threshold * 2)
            state = act.copy()
            charge = ch.copy()

            # Decode output: read 8 output bits, find closest symbol
            if p == 1:
                out_bits = charge[out_start:out_start + n_bits]
                # Score each possible output symbol
                scores = np.zeros(V, dtype=np.float32)
                for j in range(V):
                    target_bits = encode_byte8(perm[j], n_bits)
                    # Dot product similarity
                    scores[j] = np.dot(out_bits, target_bits * 2 - 1)

                probs = softmax(scores)
                if np.argmax(scores) == perm[i]:
                    correct += 1
                # Find index of perm[i] in the scoring
                target_prob += probs[perm[i]]

    acc = correct / V
    tp = target_prob / V
    return acc, 0.5 * acc + 0.5 * tp


# ============================================================
# Mutation utilities
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


# ============================================================
# 3D Mood mutation (control — tick learnable)
# ============================================================

def mutate_3d(mask, W, mood_x, mood_y, mood_z, N):
    """3 mood floats: TYPE, DEPTH (tick), INTENSITY."""
    ticks = max(1, int(1 + mood_y * 11))   # 1..12
    n_changes = max(1, int(1 + mood_z * 14))  # 1..15

    for _ in range(n_changes):
        if mood_x < 0.33:
            if random.random() < 0.7:
                add_connection(mask, W, N)
            else:
                flip_connection(mask)
        elif mood_x < 0.66:
            r = random.random()
            if r < 0.6:
                rewire_connection(mask, W, N)
            elif r < 0.8:
                flip_connection(mask)
            else:
                add_connection(mask, W, N)
        else:
            if random.random() < 0.5:
                flip_connection(mask)
            else:
                toggle_weight(mask, W)
    return ticks


# ============================================================
# 2D Mood mutation (tick NOT learnable)
# ============================================================

def mutate_2d(mask, W, mood_x, mood_z, N):
    """2D mood: specialist + intensity. Tick is NOT learnable."""
    n_changes = max(1, int(1 + mood_z * 14))  # 1..15

    for _ in range(n_changes):
        if mood_x < 0.33:
            # SCOUT
            if random.random() < 0.7:
                add_connection(mask, W, N)
            else:
                flip_connection(mask)
        elif mood_x < 0.66:
            # REWIRER
            r = random.random()
            if r < 0.6:
                rewire_connection(mask, W, N)
            elif r < 0.8:
                flip_connection(mask)
            else:
                add_connection(mask, W, N)
        else:
            # REFINER
            if random.random() < 0.5:
                flip_connection(mask)
            else:
                toggle_weight(mask, W)


# ============================================================
# Mood analysis
# ============================================================

def analyze_mood_2d(history, floor_ceil_hits):
    """Analyze 2D mood convergence."""
    if len(history) < 3:
        return {'floor_ceil': floor_ceil_hits, 'insufficient_data': True}

    mxs = [h['mx'] for h in history]
    mzs = [h['mz'] for h in history]
    n_chs = [h['n_ch'] for h in history]

    n = len(history)
    third = max(1, n // 3)

    first_mx = np.mean(mxs[:third])
    last_mx = np.mean(mxs[-third:])
    first_mz = np.mean(mzs[:third])
    last_mz = np.mean(mzs[-third:])

    scout_pct = sum(1 for m in mxs if m < 0.33) / n
    rewirer_pct = sum(1 for m in mxs if 0.33 <= m < 0.66) / n
    refiner_pct = sum(1 for m in mxs if m >= 0.66) / n

    nch_hist = {}
    for c in n_chs:
        nch_hist[c] = nch_hist.get(c, 0) + 1

    return {
        'mood_x': {'start': round(first_mx, 4), 'end': round(last_mx, 4),
                    'mean': round(np.mean(mxs), 4), 'std': round(np.std(mxs), 4)},
        'mood_z': {'start': round(first_mz, 4), 'end': round(last_mz, 4),
                    'mean': round(np.mean(mzs), 4), 'std': round(np.std(mzs), 4)},
        'phase_shift': {
            'x_delta': round(last_mx - first_mx, 4),
            'z_delta': round(last_mz - first_mz, 4),
        },
        'specialist_pct': {'scout': round(scout_pct, 4), 'rewirer': round(rewirer_pct, 4),
                           'refiner': round(refiner_pct, 4)},
        'nch_mean': round(np.mean(n_chs), 2),
        'nch_hist': {str(k): v for k, v in sorted(nch_hist.items())},
        'floor_ceil': floor_ceil_hits,
    }


def analyze_mood_3d(history, floor_ceil_hits):
    """Analyze 3D mood convergence (reused from original)."""
    if len(history) < 3:
        return {'floor_ceil': floor_ceil_hits, 'insufficient_data': True}

    mxs = [h['mx'] for h in history]
    mys = [h['my'] for h in history]
    mzs = [h['mz'] for h in history]
    ticks = [h['tick'] for h in history]
    n_chs = [h['n_ch'] for h in history]

    n = len(history)
    third = max(1, n // 3)

    first_mx, last_mx = np.mean(mxs[:third]), np.mean(mxs[-third:])
    first_my, last_my = np.mean(mys[:third]), np.mean(mys[-third:])
    first_mz, last_mz = np.mean(mzs[:third]), np.mean(mzs[-third:])

    scout_pct = sum(1 for m in mxs if m < 0.33) / n
    rewirer_pct = sum(1 for m in mxs if 0.33 <= m < 0.66) / n
    refiner_pct = sum(1 for m in mxs if m >= 0.66) / n

    tick_hist = {}
    for t in ticks:
        tick_hist[t] = tick_hist.get(t, 0) + 1
    nch_hist = {}
    for c in n_chs:
        nch_hist[c] = nch_hist.get(c, 0) + 1

    return {
        'mood_x': {'start': round(first_mx, 4), 'end': round(last_mx, 4),
                    'mean': round(np.mean(mxs), 4)},
        'mood_y': {'start': round(first_my, 4), 'end': round(last_my, 4),
                    'mean': round(np.mean(mys), 4)},
        'mood_z': {'start': round(first_mz, 4), 'end': round(last_mz, 4),
                    'mean': round(np.mean(mzs), 4)},
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
        'floor_ceil': floor_ceil_hits,
    }


# ============================================================
# Training: 2D Mood (fix eval)
# ============================================================

def train_2d_mood(seed, V, internal, max_attempts=64000, eval_tick=8):
    """Train with 2D learnable mood + FIXED eval tick."""
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal + V
    out_start = N - V
    mask, W, perm = init_network(V, N, seed)

    acc, score = eval_net(mask, W, perm, V, N, out_start, ticks=eval_tick)
    best_acc = acc

    mood_x, mood_z = 0.5, 0.5  # ONLY 2 moods (no tick!)
    mood_history = []
    floor_ceil_hits = {'x_floor': 0, 'x_ceil': 0, 'z_floor': 0, 'z_ceil': 0}

    for att in range(max_attempts):
        saved_mask = mask.copy()
        saved_W = W.copy()
        saved_mx, saved_mz = mood_x, mood_z

        # Mutate mood (20% chance per axis)
        if random.random() < 0.2:
            mood_x = float(np.clip(mood_x + random.gauss(0, 0.15), 0.0, 1.0))
        if random.random() < 0.2:
            mood_z = float(np.clip(mood_z + random.gauss(0, 0.15), 0.0, 1.0))

        # Track floor/ceil
        if mood_x <= 0.01: floor_ceil_hits['x_floor'] += 1
        if mood_x >= 0.99: floor_ceil_hits['x_ceil'] += 1
        if mood_z <= 0.01: floor_ceil_hits['z_floor'] += 1
        if mood_z >= 0.99: floor_ceil_hits['z_ceil'] += 1

        # Mutate network (guided by 2D mood)
        mutate_2d(mask, W, mood_x, mood_z, N)

        # ALWAYS eval_tick — NOT learnable!
        new_acc, new_score = eval_net(mask, W, perm, V, N, out_start, ticks=eval_tick)

        if new_score > score:
            score = new_score
            best_acc = max(best_acc, new_acc)
            mood_history.append({
                'att': att, 'mx': round(mood_x, 4), 'mz': round(mood_z, 4),
                'n_ch': max(1, int(1 + mood_z * 14)), 'acc': round(new_acc, 4)
            })
        else:
            mask[:] = saved_mask
            W[:] = saved_W
            mood_x, mood_z = saved_mx, saved_mz

        if best_acc >= 0.99:
            break

    stats = analyze_mood_2d(mood_history, floor_ceil_hits)
    stats['best_acc'] = best_acc
    stats['n_improvements'] = len(mood_history)
    stats['conns'] = int((mask != 0).sum())
    stats['eval_tick'] = eval_tick

    return best_acc, mood_history, stats


# ============================================================
# Training: 3D Mood (control — tick learnable)
# ============================================================

def train_3d_mood(seed, V, internal, max_attempts=64000):
    """Train with 3D learnable mood (tick learnable). CONTROL."""
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal + V
    out_start = N - V
    mask, W, perm = init_network(V, N, seed)

    acc, score = eval_net(mask, W, perm, V, N, out_start, ticks=8)
    best_acc = acc

    mood_x, mood_y, mood_z = 0.5, 0.5, 0.5
    mood_history = []
    floor_ceil_hits = {'x_floor': 0, 'x_ceil': 0, 'y_floor': 0, 'y_ceil': 0,
                       'z_floor': 0, 'z_ceil': 0}

    for att in range(max_attempts):
        saved_mask = mask.copy()
        saved_W = W.copy()
        saved_mx, saved_my, saved_mz = mood_x, mood_y, mood_z

        if random.random() < 0.2:
            mood_x = float(np.clip(mood_x + random.gauss(0, 0.15), 0.0, 1.0))
        if random.random() < 0.2:
            mood_y = float(np.clip(mood_y + random.gauss(0, 0.15), 0.0, 1.0))
        if random.random() < 0.2:
            mood_z = float(np.clip(mood_z + random.gauss(0, 0.15), 0.0, 1.0))

        if mood_x <= 0.01: floor_ceil_hits['x_floor'] += 1
        if mood_x >= 0.99: floor_ceil_hits['x_ceil'] += 1
        if mood_y <= 0.01: floor_ceil_hits['y_floor'] += 1
        if mood_y >= 0.99: floor_ceil_hits['y_ceil'] += 1
        if mood_z <= 0.01: floor_ceil_hits['z_floor'] += 1
        if mood_z >= 0.99: floor_ceil_hits['z_ceil'] += 1

        ticks = mutate_3d(mask, W, mood_x, mood_y, mood_z, N)
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
            mask[:] = saved_mask
            W[:] = saved_W
            mood_x, mood_y, mood_z = saved_mx, saved_my, saved_mz

        if best_acc >= 0.99:
            break

    stats = analyze_mood_3d(mood_history, floor_ceil_hits)
    stats['best_acc'] = best_acc
    stats['n_improvements'] = len(mood_history)
    stats['conns'] = int((mask != 0).sum())

    return best_acc, mood_history, stats


# ============================================================
# Training: Baseline (refiner only, tick=8)
# ============================================================

def train_baseline(seed, V, internal, max_attempts=64000):
    """Classic refiner-only hill climbing. tick=8 fixed."""
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal + V
    out_start = N - V
    mask, W, perm = init_network(V, N, seed)

    acc, score = eval_net(mask, W, perm, V, N, out_start, ticks=8)
    best_acc = acc
    stale = 0

    for att in range(max_attempts):
        saved_mask = mask.copy()
        saved_W = W.copy()

        if stale < max_attempts // 3:
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
# Training: Byte_8 encoding variant
# ============================================================

def train_baseline_byte8(seed, V, internal_total, max_attempts=8000):
    """Baseline with byte_8 encoding. 8-bit I/O, more internal neurons."""
    np.random.seed(seed)
    random.seed(seed)

    input_size = 8   # 8 bits for input
    output_size = 8  # 8 bits for output
    N = input_size + internal_total + output_size
    out_start = N - output_size

    r = np.random.rand(N, N)
    mask = np.zeros((N, N), dtype=np.float32)
    mask[r < 0.03] = -1
    mask[r > 0.97] = 1
    np.fill_diagonal(mask, 0)
    W = np.where(np.random.rand(N, N) > 0.5, np.float32(0.5), np.float32(1.5))

    perm = np.random.permutation(V)

    acc, score = eval_net_byte8(mask, W, perm, V, N, input_size, output_size,
                                out_start, ticks=8)
    best_acc = acc
    stale = 0

    for att in range(max_attempts):
        saved_mask = mask.copy()
        saved_W = W.copy()

        if stale < max_attempts // 3:
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

        new_acc, new_score = eval_net_byte8(mask, W, perm, V, N, input_size,
                                           output_size, out_start, ticks=8)

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
# Training: One-hot baseline (standard, for SQ1 comparison)
# ============================================================

def train_baseline_onehot(seed, V, internal, max_attempts=8000):
    """Standard one-hot baseline for encoding comparison."""
    return train_baseline(seed, V, internal, max_attempts)


# ============================================================
# Side Quest 3: Context-length with byte_8 (bigram)
# ============================================================

def make_bigram_data():
    """Create English bigram frequency table → permutation-like task.
    Top 64 bigrams as a lookup task."""
    # Common English bigrams (approximate top frequency order)
    bigrams = [
        'th', 'he', 'in', 'er', 'an', 'on', 'en', 're',
        'ed', 'nd', 'ha', 'at', 'ou', 'to', 'it', 'is',
        'st', 'or', 'es', 'te', 'of', 'le', 'se', 'al',
        'ar', 'ni', 'ne', 'ng', 'ti', 'io', 'de', 'co',
        'ri', 'li', 'ra', 'ea', 'ce', 'no', 'ta', 'as',
        'el', 'me', 'ro', 'di', 'ic', 've', 'la', 'us',
        'ch', 'ge', 'ma', 'si', 'ur', 'pe', 'na', 'un',
        'om', 'lo', 'ac', 'ad', 'ca', 'be', 'ab', 'am',
    ]
    return bigrams[:64]


def eval_context_byte8(mask, W, bigrams, context_len, N, input_size,
                       output_size, out_start, ticks=8, threshold=0.5, leak=0.85):
    """Eval bigram prediction with variable context length.
    Input: byte_8 encoding of context_len characters.
    Output: byte_8 of the NEXT character."""
    state = np.zeros(N, dtype=np.float32)
    charge = np.zeros(N, dtype=np.float32)
    Weff = W * mask
    correct = 0
    n_bits = 8
    total = len(bigrams)

    for p in range(2):
        for idx, bg in enumerate(bigrams):
            # Encode input chars
            world = np.zeros(input_size, dtype=np.float32)
            for c_idx in range(min(context_len, len(bg) - 1)):
                # Last context_len chars before the target
                char_val = ord(bg[c_idx]) - ord('a')
                bits = encode_byte8(char_val, n_bits)
                offset = c_idx * n_bits
                if offset + n_bits <= input_size:
                    world[offset:offset + n_bits] = bits

            act = state.copy()
            ch = charge.copy()
            for t in range(ticks):
                if t == 0:
                    act[:input_size] = world
                raw = act @ Weff + act * 0.1
                ch += raw * 0.3
                ch *= leak
                act = np.maximum(ch - threshold, 0.0)
                ch = np.clip(ch, -threshold * 2, threshold * 2)
            state = act.copy()
            charge = ch.copy()

            if p == 1:
                # Target: last character
                target_val = ord(bg[-1]) - ord('a')
                out_bits = charge[out_start:out_start + n_bits]
                # Score each letter a-z
                best_score = -999
                best_char = 0
                for cv in range(26):
                    tb = encode_byte8(cv, n_bits)
                    s = np.dot(out_bits, tb * 2 - 1)
                    if s > best_score:
                        best_score = s
                        best_char = cv
                if best_char == target_val:
                    correct += 1

    return correct / total


def train_context_byte8(seed, context_len, max_attempts=8000):
    """Train bigram prediction with given context length."""
    np.random.seed(seed)
    random.seed(seed)

    n_bits = 8
    input_size = context_len * n_bits
    output_size = n_bits
    internal = 192 - input_size - output_size
    if internal < 32:
        internal = 32
    N = input_size + internal + output_size
    out_start = N - output_size

    bigrams = make_bigram_data()

    r = np.random.rand(N, N)
    mask = np.zeros((N, N), dtype=np.float32)
    mask[r < 0.03] = -1
    mask[r > 0.97] = 1
    np.fill_diagonal(mask, 0)
    W = np.where(np.random.rand(N, N) > 0.5, np.float32(0.5), np.float32(1.5))

    acc = eval_context_byte8(mask, W, bigrams, context_len, N, input_size,
                             output_size, out_start, ticks=8)
    best_acc = acc
    score = acc  # Simple accuracy scoring
    stale = 0

    for att in range(max_attempts):
        saved_mask = mask.copy()
        saved_W = W.copy()

        if stale < max_attempts // 3:
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

        new_acc = eval_context_byte8(mask, W, bigrams, context_len, N, input_size,
                                     output_size, out_start, ticks=8)

        if new_acc > score:
            score = new_acc
            best_acc = max(best_acc, new_acc)
            stale = 0
        else:
            mask[:] = saved_mask
            W[:] = saved_W
            stale += 1

        if best_acc >= 0.99:
            break

    return best_acc


# ============================================================
# Parallel worker wrappers
# ============================================================

def worker_2d_mood(args):
    seed, V, internal, max_att, eval_tick = args
    try:
        best_acc, history, stats = train_2d_mood(seed, V, internal, max_att, eval_tick)
        return {'seed': seed, 'acc': best_acc, 'stats': stats,
                'history_len': len(history), 'history_sample': history[-5:] if history else [],
                'eval_tick': eval_tick}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}


def worker_3d_mood(args):
    seed, V, internal, max_att = args
    try:
        best_acc, history, stats = train_3d_mood(seed, V, internal, max_att)
        return {'seed': seed, 'acc': best_acc, 'stats': stats,
                'history_len': len(history), 'history_sample': history[-5:] if history else []}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}


def worker_baseline(args):
    seed, V, internal, max_att = args
    try:
        best_acc, conns = train_baseline(seed, V, internal, max_att)
        return {'seed': seed, 'acc': best_acc, 'conns': conns}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}


def worker_byte8(args):
    seed, V, internal_total, max_att = args
    try:
        best_acc, conns = train_baseline_byte8(seed, V, internal_total, max_att)
        return {'seed': seed, 'acc': best_acc, 'conns': conns}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}


def worker_onehot(args):
    seed, V, internal, max_att = args
    try:
        best_acc, conns = train_baseline_onehot(seed, V, internal, max_att)
        return {'seed': seed, 'acc': best_acc, 'conns': conns}
    except Exception as e:
        return {'seed': seed, 'acc': 0, 'error': str(e), 'tb': traceback.format_exc()}


def worker_context(args):
    seed, context_len, max_att = args
    try:
        best_acc = train_context_byte8(seed, context_len, max_att)
        return {'seed': seed, 'context_len': context_len, 'acc': best_acc}
    except Exception as e:
        return {'seed': seed, 'context_len': context_len, 'acc': 0,
                'error': str(e), 'tb': traceback.format_exc()}


def worker_2d_calibration(args):
    seed, V, internal, max_att, eval_tick = args
    try:
        best_acc, history, stats = train_2d_mood(seed, V, internal, max_att, eval_tick)
        return {'seed': seed, 'V': V, 'acc': best_acc, 'stats': stats}
    except Exception as e:
        return {'seed': seed, 'V': V, 'acc': 0,
                'error': str(e), 'tb': traceback.format_exc()}


# ============================================================
# Log helpers
# ============================================================

def log_condition_results(label, res_list, elapsed):
    """Log per-seed results and summary for a condition."""
    accs = [r['acc'] for r in res_list]
    for r in res_list:
        err = f" ERROR: {r['error']}" if 'error' in r else ""
        log_progress(f"    seed={r['seed']}: acc={r['acc']:.1%}{err}")
    log_progress(f"  {label}: avg={np.mean(accs):.1%} best={max(accs):.1%} time={elapsed:.1f}s")
    return accs


def log_mood_2d_stats(res_list):
    """Log 2D mood convergence details."""
    for r in res_list:
        if 'stats' in r and 'mood_x' in r.get('stats', {}):
            s = r['stats']
            spec = s.get('specialist_pct', {})
            ps = s.get('phase_shift', {})
            log_progress(f"    seed={r['seed']} mood: "
                         f"x={s['mood_x']['mean']:.2f}({s['mood_x']['start']:.2f}->{s['mood_x']['end']:.2f}) "
                         f"z={s['mood_z']['mean']:.2f}({s['mood_z']['start']:.2f}->{s['mood_z']['end']:.2f}) "
                         f"scout={spec.get('scout', 0):.0%} rewirer={spec.get('rewirer', 0):.0%} "
                         f"refiner={spec.get('refiner', 0):.0%} "
                         f"nch_avg={s.get('nch_mean', '?')} "
                         f"improvements={s.get('n_improvements', '?')}")
            log_progress(f"             floor/ceil: {s.get('floor_ceil', {})}")


def log_mood_3d_stats(res_list):
    """Log 3D mood convergence details."""
    for r in res_list:
        if 'stats' in r and 'mood_x' in r.get('stats', {}):
            s = r['stats']
            spec = s.get('specialist_pct', {})
            ps = s.get('phase_shift', {})
            log_progress(f"    seed={r['seed']} mood: "
                         f"x={s['mood_x']['mean']:.2f} y={s['mood_y']['mean']:.2f} "
                         f"z={s['mood_z']['mean']:.2f} "
                         f"x_shift={ps.get('x_delta', '?'):+.3f} "
                         f"y_shift={ps.get('y_delta', '?'):+.3f} "
                         f"z_shift={ps.get('z_delta', '?'):+.3f} "
                         f"tick_avg={s.get('tick_mean', '?')} "
                         f"improvements={s.get('n_improvements', '?')}")
            log_progress(f"             floor/ceil: {s.get('floor_ceil', {})}")


# ============================================================
# MAIN TEST: 2D Mood V=64, 64K
# ============================================================

def run_main_test():
    """Main test: baseline vs 3d_mood vs 2d_mood_fix8 vs 2d_mood_fix4."""
    V = 64
    INTERNAL = 128  # N=192 standard (64+128+64 for one-hot)
    MAX_ATT = 64000

    log_progress("=" * 70)
    log_progress("MAIN TEST: 2D Mood Fix Eval | V=64 | N=192 | 64K attempts")
    log_progress("=" * 70)

    phase_results = {}

    # --- 1. BASELINE (refiner only, tick=8) ---
    label = "baseline_V64_64K"
    log_progress(f"\n  [1/4] START: {label}")
    t0 = time.time()
    args = [(s, V, INTERNAL, MAX_ATT) for s in SEEDS]
    with Pool(N_WORKERS) as pool:
        base_res = pool.map(worker_baseline, args)
    elapsed = time.time() - t0
    accs = log_condition_results(label, base_res, elapsed)
    phase_results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'details': base_res
    }
    results[label] = phase_results[label]
    save_results()
    time.sleep(2)

    # --- 2. 3D MOOD (control — tick learnable) ---
    label = "3d_mood_V64_64K"
    log_progress(f"\n  [2/4] START: {label}")
    t0 = time.time()
    args = [(s, V, INTERNAL, MAX_ATT) for s in SEEDS]
    with Pool(N_WORKERS) as pool:
        mood3d_res = pool.map(worker_3d_mood, args)
    elapsed = time.time() - t0
    accs = log_condition_results(label, mood3d_res, elapsed)
    log_mood_3d_stats(mood3d_res)
    phase_results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'details': mood3d_res
    }
    results[label] = phase_results[label]
    save_results()
    time.sleep(2)

    # --- 3. 2D MOOD FIX TICK=8 (THE QUESTION) ---
    label = "2d_mood_fix8_V64_64K"
    log_progress(f"\n  [3/4] START: {label} ← THE KEY TEST")
    t0 = time.time()
    args = [(s, V, INTERNAL, MAX_ATT, 8) for s in SEEDS]
    with Pool(N_WORKERS) as pool:
        mood2d_8_res = pool.map(worker_2d_mood, args)
    elapsed = time.time() - t0
    accs = log_condition_results(label, mood2d_8_res, elapsed)
    log_mood_2d_stats(mood2d_8_res)
    phase_results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'details': mood2d_8_res
    }
    results[label] = phase_results[label]
    save_results()
    time.sleep(2)

    # --- 4. 2D MOOD FIX TICK=4 (faster eval) ---
    label = "2d_mood_fix4_V64_64K"
    log_progress(f"\n  [4/4] START: {label}")
    t0 = time.time()
    args = [(s, V, INTERNAL, MAX_ATT, 4) for s in SEEDS]
    with Pool(N_WORKERS) as pool:
        mood2d_4_res = pool.map(worker_2d_mood, args)
    elapsed = time.time() - t0
    accs = log_condition_results(label, mood2d_4_res, elapsed)
    log_mood_2d_stats(mood2d_4_res)
    phase_results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'details': mood2d_4_res
    }
    results[label] = phase_results[label]
    save_results()

    # Summary
    log_progress("\n" + "=" * 70)
    log_progress("MAIN TEST SUMMARY")
    log_progress("=" * 70)
    log_progress(f"{'Method':<30} {'Avg Acc':>10} {'Best Acc':>10} {'Time':>10}")
    log_progress("-" * 70)
    for k in ['baseline_V64_64K', '3d_mood_V64_64K', '2d_mood_fix8_V64_64K', '2d_mood_fix4_V64_64K']:
        r = phase_results[k]
        log_progress(f"{k:<30} {r['avg_acc']:>9.1%} {r['best_acc']:>9.1%} {r['time']:>9.1f}s")

    return phase_results


# ============================================================
# SIDE QUEST 1: Byte vs One-Hot encoding
# ============================================================

def run_side_quest_1():
    """Byte_8 vs one-hot encoding, V=64, 8K attempts, 5 seeds."""
    V = 64
    MAX_ATT = 8000

    log_progress("\n" + "=" * 70)
    log_progress("SIDE QUEST 1: Byte_8 vs One-Hot Encoding | V=64 | 8K attempts")
    log_progress("  one_hot: 64 I/O + 128 internal = N=192 (64+128+64)")
    log_progress("  byte_8:  8 I/O + 176 internal = N=192 (8+176+8)")
    log_progress("=" * 70)

    sq1_results = {}

    # --- ONE-HOT ---
    label = "sq1_onehot_V64_8K"
    log_progress(f"\n  START: {label}")
    t0 = time.time()
    args = [(s, V, 128, MAX_ATT) for s in SEEDS]  # 64+128+64=256... wait
    # N=192 means: V=64 one-hot → 64 in + 64 internal + 64 out = 192
    # Actually: INTERNAL = 192 - 2*V = 192 - 128 = 64
    internal_onehot = 192 - 2 * V  # = 64
    args = [(s, V, internal_onehot, MAX_ATT) for s in SEEDS]
    with Pool(N_WORKERS) as pool:
        onehot_res = pool.map(worker_onehot, args)
    elapsed = time.time() - t0
    accs = log_condition_results(label, onehot_res, elapsed)
    log_progress(f"  Layout: {V} in + {internal_onehot} internal + {V} out = {V + internal_onehot + V}")
    sq1_results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'layout': f'{V}+{internal_onehot}+{V}={V+internal_onehot+V}'
    }
    results[label] = sq1_results[label]
    save_results()
    time.sleep(2)

    # --- BYTE_8 ---
    label = "sq1_byte8_V64_8K"
    log_progress(f"\n  START: {label}")
    t0 = time.time()
    # byte_8: 8 in + 176 internal + 8 out = 192
    internal_byte8 = 192 - 8 - 8  # = 176
    args = [(s, V, internal_byte8, MAX_ATT) for s in SEEDS]
    with Pool(N_WORKERS) as pool:
        byte8_res = pool.map(worker_byte8, args)
    elapsed = time.time() - t0
    accs = log_condition_results(label, byte8_res, elapsed)
    log_progress(f"  Layout: 8 in + {internal_byte8} internal + 8 out = {8 + internal_byte8 + 8}")
    sq1_results[label] = {
        'accs': accs, 'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
        'time': elapsed, 'layout': f'8+{internal_byte8}+8={8+internal_byte8+8}'
    }
    results[label] = sq1_results[label]
    save_results()

    # Summary
    log_progress("\n  SQ1 SUMMARY:")
    for k, v in sq1_results.items():
        log_progress(f"    {k}: avg={v['avg_acc']:.1%} best={v['best_acc']:.1%} layout={v['layout']}")

    return sq1_results


# ============================================================
# SIDE QUEST 2: Intrinsic Dimension Calibration
# ============================================================

def run_side_quest_2():
    """Mood_z convergence across V=16/32/64 with fix tick=8."""
    MAX_ATT = 8000

    log_progress("\n" + "=" * 70)
    log_progress("SIDE QUEST 2: Intrinsic Dimension Calibration via mood_z")
    log_progress("  Fix tick=8, measure mood_z convergence across V sizes")
    log_progress("  V=16 N=64 | V=32 N=96 | V=64 N=192")
    log_progress("=" * 70)

    configs = [
        (16, 32),    # V=16, internal=32  → N=64
        (32, 32),    # V=32, internal=32  → N=96
        (64, 64),    # V=64, internal=64  → N=192
    ]

    sq2_results = {}

    for V, internal in configs:
        N_total = V + internal + V
        label = f"sq2_V{V}_N{N_total}_8K"
        log_progress(f"\n  START: {label}")
        t0 = time.time()
        args = [(s, V, internal, MAX_ATT, 8) for s in SEEDS]
        with Pool(N_WORKERS) as pool:
            cal_res = pool.map(worker_2d_calibration, args)
        elapsed = time.time() - t0

        accs = [r['acc'] for r in cal_res]
        mz_ends = []
        for r in cal_res:
            err = f" ERROR: {r.get('error', '')}" if 'error' in r else ""
            s = r.get('stats', {})
            mz_info = s.get('mood_z', {})
            mz_end = mz_info.get('end', '?')
            mz_mean = mz_info.get('mean', '?')
            if isinstance(mz_end, (int, float)):
                mz_ends.append(mz_end)
            log_progress(f"    seed={r['seed']}: acc={r['acc']:.1%} "
                         f"mz_end={mz_end} mz_mean={mz_mean}{err}")

        avg_mz_end = np.mean(mz_ends) if mz_ends else 0
        log_progress(f"  {label}: avg_acc={np.mean(accs):.1%} avg_mz_end={avg_mz_end:.3f}")

        sq2_results[label] = {
            'V': V, 'N': N_total, 'accs': accs,
            'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
            'avg_mz_end': float(avg_mz_end),
            'time': elapsed, 'details': cal_res
        }
        results[label] = sq2_results[label]
        save_results()
        time.sleep(2)

    # Summary
    log_progress("\n  SQ2 SUMMARY — mood_z convergence vs V:")
    log_progress(f"  {'Config':<25} {'Avg Acc':>10} {'avg_mz_end':>12} {'Interpretation'}")
    log_progress("  " + "-" * 65)
    for k, v in sq2_results.items():
        interp = "EASY" if v['avg_mz_end'] < 0.3 else "MEDIUM" if v['avg_mz_end'] < 0.5 else "HARD"
        log_progress(f"  {k:<25} {v['avg_acc']:>9.1%} {v['avg_mz_end']:>11.3f}  {interp}")

    return sq2_results


# ============================================================
# SIDE QUEST 3: Context length with byte_8
# ============================================================

def run_side_quest_3():
    """Bigram prediction with 1/2/3 char context, byte_8 encoding."""
    MAX_ATT = 8000

    log_progress("\n" + "=" * 70)
    log_progress("SIDE QUEST 3: Context Length with byte_8 Encoding")
    log_progress("  English bigram prediction, 8K attempts, 3 seeds")
    log_progress("  1 char (8 neuron input) vs 2 char (16) vs 3 char (24)")
    log_progress("=" * 70)

    sq3_results = {}

    for ctx_len in [1, 2, 3]:
        input_size = ctx_len * 8
        internal = max(32, 192 - input_size - 8)
        N_total = input_size + internal + 8
        label = f"sq3_ctx{ctx_len}_byte8_8K"
        log_progress(f"\n  START: {label} (input={input_size}, internal={internal}, N={N_total})")
        t0 = time.time()
        args = [(s, ctx_len, MAX_ATT) for s in SEEDS_SHORT]
        with Pool(min(N_WORKERS, len(SEEDS_SHORT))) as pool:
            ctx_res = pool.map(worker_context, args)
        elapsed = time.time() - t0

        accs = [r['acc'] for r in ctx_res]
        for r in ctx_res:
            err = f" ERROR: {r.get('error', '')}" if 'error' in r else ""
            log_progress(f"    seed={r['seed']}: acc={r['acc']:.1%}{err}")
        log_progress(f"  {label}: avg={np.mean(accs):.1%} best={max(accs):.1%} time={elapsed:.1f}s")

        sq3_results[label] = {
            'context_len': ctx_len, 'accs': accs,
            'avg_acc': float(np.mean(accs)), 'best_acc': float(max(accs)),
            'time': elapsed, 'layout': f'{input_size}+{internal}+8={N_total}'
        }
        results[label] = sq3_results[label]
        save_results()
        time.sleep(2)

    # Summary
    log_progress("\n  SQ3 SUMMARY — Context length effect with byte_8:")
    for k, v in sq3_results.items():
        log_progress(f"    {k}: avg={v['avg_acc']:.1%} best={v['best_acc']:.1%} layout={v['layout']}")

    return sq3_results


# ============================================================
# MAIN
# ============================================================

def main():
    log_progress("=" * 70)
    log_progress("2D MOOD FIX EVAL TEST — STARTED")
    log_progress(f"Seeds: {SEEDS}, Workers: {N_WORKERS}, CPUs: {cpu_count()}")
    log_progress("=" * 70)

    # PRIORITY 1: Main test (2D mood V=64 64K) — MOST IMPORTANT
    try:
        main_results = run_main_test()
    except Exception as e:
        log_progress(f"MAIN TEST CRASH: {traceback.format_exc()}")
    time.sleep(2)

    # PRIORITY 2: Side Quest 1 (byte vs one-hot) — quick, 8K
    try:
        sq1 = run_side_quest_1()
    except Exception as e:
        log_progress(f"SQ1 CRASH: {traceback.format_exc()}")
    time.sleep(2)

    # PRIORITY 3: Side Quest 2 (d calibration) — quick, 8K
    try:
        sq2 = run_side_quest_2()
    except Exception as e:
        log_progress(f"SQ2 CRASH: {traceback.format_exc()}")
    time.sleep(2)

    # PRIORITY 4: Side Quest 3 (context byte_8) — if time permits
    try:
        sq3 = run_side_quest_3()
    except Exception as e:
        log_progress(f"SQ3 CRASH: {traceback.format_exc()}")

    # FINAL SUMMARY
    log_progress("\n" + "=" * 70)
    log_progress("ALL TESTS COMPLETE — FINAL SUMMARY")
    log_progress("=" * 70)
    for k, v in results.items():
        if isinstance(v, dict) and 'avg_acc' in v:
            extra = ""
            if 'layout' in v:
                extra = f" layout={v['layout']}"
            if 'avg_mz_end' in v:
                extra = f" mz_end={v['avg_mz_end']:.3f}"
            log_progress(f"  {k}: avg={v['avg_acc']:.1%} best={v['best_acc']:.1%}{extra}")

    log_progress("\nDONE.")


if __name__ == '__main__':
    main()
