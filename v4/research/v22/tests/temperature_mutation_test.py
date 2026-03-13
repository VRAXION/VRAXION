"""
Temperature-Modulated Mutation Test — v22 SelfWiringGraph
==========================================================
Neuromodulation-inspired adaptive mutation: temperature controls
the TYPE and SIZE of mutations based on learning progress.

- Improving -> cool down (exploitation, fine-tuning)
- Stagnating -> heat up (exploration, large jumps)

Temperature zones:
  < 0.5   FOCUSED: flip only (finest, most effective)
  0.5-1.5 NORMAL: standard mixed mutation
  1.5-3.0 WIDE: block mutation (4-5 changes at once)
  > 3.0   EARTHQUAKE: massive rewire + region flip

Phase A: 4 modes x 3 tasks x 5 seeds = 60 jobs (synthetic)
Phase B: local temperature (if Phase A positive), 64-class only
Phase C: English bigram (if Phase A/B positive)
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
#  English text corpus (reused from cht_regrowth_test.py)
# ============================================================

TEXT = """the quick brown fox jumps over the lazy dog.
the cat sat on the mat. the dog chased the cat around the garden.
she sells sea shells by the sea shore. peter piper picked a peck of pickled peppers.
to be or not to be that is the question. all that glitters is not gold.
a stitch in time saves nine. the early bird catches the worm.
actions speak louder than words. practice makes perfect.
knowledge is power. time is money. better late than never.
the pen is mightier than the sword. where there is a will there is a way.
an apple a day keeps the doctor away. birds of a feather flock together.
every cloud has a silver lining. fortune favors the bold.
the best things in life are free. honesty is the best policy.
if at first you do not succeed try try again. rome was not built in a day.
the grass is always greener on the other side. curiosity killed the cat.
do not count your chickens before they hatch. a penny saved is a penny earned.
two wrongs do not make a right. when in rome do as the romans do.
the squeaky wheel gets the grease. you can not judge a book by its cover.
beauty is in the eye of the beholder. absence makes the heart grow fonder.
the journey of a thousand miles begins with a single step.
where there is smoke there is fire. still waters run deep.
a rolling stone gathers no moss. look before you leap.
necessity is the mother of invention. blood is thicker than water.
the apple does not fall far from the tree. there is no place like home.
you can lead a horse to water but you can not make it drink.
every dog has its day. do not put all your eggs in one basket.""".lower()

chars = sorted(set(TEXT))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
VOCAB = len(chars)


def compute_bigram_dist():
    counts = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for i in range(len(TEXT) - 1):
        a, b = TEXT[i], TEXT[i + 1]
        if a in char_to_idx and b in char_to_idx:
            counts[char_to_idx[a], char_to_idx[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


BIGRAM_DIST = compute_bigram_dist()
ACTIVE_INPUTS = [i for i in range(VOCAB) if BIGRAM_DIST[i].sum() > 0.01]


# ============================================================
#  Temperature-modulated mutation
# ============================================================

MAX_DENSITY = 0.15


def flip_single(net):
    """Flip sign of a single random existing connection."""
    alive = np.argwhere(net.mask != 0)
    if len(alive) == 0:
        return
    idx = alive[random.randint(0, len(alive) - 1)]
    net.mask[int(idx[0]), int(idx[1])] *= -1


def capped_mutate_once(net, rate=0.03):
    """Single mutation op, density-aware (no add if over cap)."""
    N = net.N
    density = (net.mask != 0).sum() / (N * (N - 1))

    if density >= MAX_DENSITY:
        # Only flip, remove, rewire
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


def temp_mutate(net, temperature):
    """Temperature-modulated mutation."""
    if temperature < 0.5:
        # FOCUSED: flip only
        flip_single(net)

    elif temperature < 1.5:
        # NORMAL: standard mixed
        r = random.random()
        if r < 0.3:
            flip_single(net)
        else:
            capped_mutate_once(net, 0.05)

    elif temperature < 3.0:
        # WIDE: block mutation
        n_changes = int(2 + temperature)
        for _ in range(n_changes):
            if random.random() < 0.5:
                flip_single(net)
            else:
                capped_mutate_once(net, 0.03)

    else:
        # EARTHQUAKE: massive rewire + region flip
        n_changes = int(temperature * 2)
        for _ in range(n_changes):
            capped_mutate_once(net, 0.03)
        # Region flip
        N = net.N
        region = random.sample(range(N), min(10, N))
        for n in region:
            alive = np.argwhere(net.mask[n] != 0).flatten()
            if len(alive) > 0:
                idx = alive[random.randint(0, len(alive) - 1)]
                net.mask[n, idx] *= -1


# ============================================================
#  Scoring functions
# ============================================================

def score_combined_synthetic(net, targets, vocab, ticks=8):
    """Combined scoring: 0.5*acc + 0.5*target_prob."""
    net.reset()
    correct = 0
    total_score = 0.0
    n = vocab
    for p in range(2):
        for inp in range(n):
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
    acc = correct / n
    net.last_acc = acc
    return total_score / n, acc


def score_combined_synthetic_perclass(net, targets, vocab, ticks=8):
    """Combined scoring + per-class accuracy for local temperature."""
    net.reset()
    correct = 0
    total_score = 0.0
    per_class = np.zeros(vocab, dtype=np.float32)
    n = vocab
    for p in range(2):
        for inp in range(n):
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
                per_class[inp] = acc_i
    acc = correct / n
    net.last_acc = acc
    return total_score / n, acc, per_class


def score_hybrid_bigram(net, true_dist, vocab, active_inputs, ticks=8):
    """Hybrid scoring for bigram: 0.5*top1 + 0.5*(1 - MSE/0.01)."""
    total_mse = 0.0
    top1_match = 0
    for inp in active_inputs:
        net.reset()
        world = np.zeros(vocab, dtype=np.float32)
        world[inp] = 1.0
        logits = net.forward(world, ticks)
        pred = softmax(logits[:vocab])
        mse = np.mean((pred - true_dist[inp]) ** 2)
        total_mse += mse
        if np.argmax(pred) == np.argmax(true_dist[inp]):
            top1_match += 1
    mean_mse = total_mse / len(active_inputs)
    top1_acc = top1_match / len(active_inputs)
    mse_score = 1.0 - mean_mse / 0.01
    return 0.5 * top1_acc + 0.5 * mse_score


def eval_bigram_metrics(net, true_dist, vocab, active_inputs, ticks=8):
    """Full bigram metrics for reporting."""
    total_mse = 0.0
    top1 = 0
    top3 = 0
    for inp in active_inputs:
        net.reset()
        world = np.zeros(vocab, dtype=np.float32)
        world[inp] = 1.0
        logits = net.forward(world, ticks)
        pred = softmax(logits[:vocab])
        target = true_dist[inp]
        total_mse += np.mean((pred - target) ** 2)
        if np.argmax(pred) == np.argmax(target):
            top1 += 1
        if np.argmax(target) in set(np.argsort(-pred)[:3]):
            top3 += 1
    n = len(active_inputs)
    return {'mse': total_mse / n, 'top1': top1 / n, 'top3': top3 / n}


# ============================================================
#  Training — Phase A: global temperature modes
# ============================================================

def train_synthetic_temp(label, n_neurons, seed, temp_mode, n_classes,
                         max_attempts=8000, ticks=8):
    """Train synthetic task with temperature-modulated mutation."""
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    net = SelfWiringGraph(n_neurons, V)
    perm = np.random.permutation(V)

    score, acc = score_combined_synthetic(net, perm, V, ticks)
    best_score = score
    best_acc = acc
    kept = 0
    stale = 0
    temperature = 1.0
    phase = "STRUCTURE"
    temp_log = []

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if temp_mode == 'baseline':
            # Standard v22 behavior with density cap for fair comparison
            if phase == "STRUCTURE":
                capped_mutate_once(net, 0.05)
            else:
                if random.random() < 0.3:
                    capped_mutate_once(net, 0.02)
                else:
                    net.mutate_weights()
        else:
            temp_mutate(net, temperature)

        new_score, new_acc = score_combined_synthetic(net, perm, V, ticks)

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)

            # Temperature cooling
            if temp_mode == 'temp_gradual':
                temperature = max(0.3, temperature * 0.95)
            elif temp_mode == 'temp_aggressive':
                temperature = max(0.2, temperature * 0.90)
            elif temp_mode == 'temp_step':
                temperature = max(0.5, temperature * 0.95)
        else:
            net.restore_state(state)
            stale += 1

            # Temperature heating
            if temp_mode == 'temp_gradual':
                if stale % 200 == 0:
                    temperature = min(5.0, temperature * 1.3)
            elif temp_mode == 'temp_aggressive':
                if stale % 200 == 0:
                    temperature = min(8.0, temperature * 1.5)
            elif temp_mode == 'temp_step':
                if stale == 1000:
                    temperature = 2.0
                elif stale == 2000:
                    temperature = 4.0
                elif stale == 3000:
                    temperature = 8.0

        # Phase transition (baseline only)
        if temp_mode == 'baseline' and phase == "STRUCTURE" and stale > 2500:
            phase = "BOTH"
            stale = 0

        if att % 1000 == 0:
            temp_log.append(temperature)

        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100

    return {
        'label': label,
        'seed': seed,
        'temp_mode': temp_mode,
        'n_classes': n_classes,
        'final_acc': best_acc,
        'accept_rate': accept_rate,
        'kept': kept,
        'attempts': att + 1,
        'conns': net.count_connections(),
        'time': elapsed,
        'final_temp': temperature,
        'temp_log': temp_log,
    }


# ============================================================
#  Training — Phase B: local temperature
# ============================================================

def train_synthetic_local(label, n_neurons, seed, local_mode, n_classes,
                          max_attempts=8000, ticks=8):
    """Train 64-class with local (per-class) temperature."""
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    net = SelfWiringGraph(n_neurons, V)
    perm = np.random.permutation(V)

    score, acc, per_class_acc = score_combined_synthetic_perclass(
        net, perm, V, ticks)
    best_score = score
    best_acc = acc
    kept = 0
    stale = 0
    global_temp = 1.0
    class_temps = np.ones(V, dtype=np.float32)

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if local_mode == 'global_temp':
            # Just global temperature (Phase A winner behavior)
            temp_mutate(net, global_temp)

        elif local_mode == 'local_temp':
            # Per-class targeted mutation
            probs = class_temps / class_temps.sum()
            target_class = np.random.choice(V, p=probs)
            target_neuron = target_class
            connected = np.argwhere(net.mask[:, target_neuron] != 0).flatten()

            if len(connected) > 0 and random.random() < 0.7:
                src = connected[random.randint(0, len(connected) - 1)]
                if class_temps[target_class] > 2.0:
                    net.mask[src, target_neuron] *= -1
                else:
                    net.W[src, target_neuron] = (
                        np.float32(1.5) if net.W[src, target_neuron] < 1.0
                        else np.float32(0.5))
            else:
                capped_mutate_once(net, 0.03)

        elif local_mode == 'hybrid':
            # 50% global temp_mutate + 50% local targeted
            if random.random() < 0.5:
                temp_mutate(net, global_temp)
            else:
                probs = class_temps / class_temps.sum()
                target_class = np.random.choice(V, p=probs)
                target_neuron = target_class
                connected = np.argwhere(
                    net.mask[:, target_neuron] != 0).flatten()
                if len(connected) > 0:
                    src = connected[random.randint(0, len(connected) - 1)]
                    if class_temps[target_class] > 2.0:
                        net.mask[src, target_neuron] *= -1
                    else:
                        net.W[src, target_neuron] = (
                            np.float32(1.5) if net.W[src, target_neuron] < 1.0
                            else np.float32(0.5))
                else:
                    capped_mutate_once(net, 0.03)

        new_score, new_acc, new_per_class = score_combined_synthetic_perclass(
            net, perm, V, ticks)

        if new_score > score:
            score = new_score
            per_class_acc = new_per_class
            kept += 1
            stale = 0
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
            global_temp = max(0.3, global_temp * 0.95)
        else:
            net.restore_state(state)
            stale += 1
            if stale % 200 == 0:
                global_temp = min(5.0, global_temp * 1.3)

        # Update per-class temperatures every 100 steps
        if att % 100 == 0:
            for i in range(V):
                if per_class_acc[i] > 0.8:
                    class_temps[i] = max(0.3, class_temps[i] * 0.95)
                elif per_class_acc[i] < 0.3:
                    class_temps[i] = min(5.0, class_temps[i] * 1.1)

        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100

    return {
        'label': label,
        'seed': seed,
        'local_mode': local_mode,
        'n_classes': n_classes,
        'final_acc': best_acc,
        'accept_rate': accept_rate,
        'kept': kept,
        'attempts': att + 1,
        'conns': net.count_connections(),
        'time': elapsed,
    }


# ============================================================
#  Training — Phase C: English bigram
# ============================================================

def train_bigram_temp(label, n_neurons, seed, temp_mode,
                      max_attempts=8000, ticks=8):
    """Train bigram with temperature-modulated mutation."""
    np.random.seed(seed)
    random.seed(seed)

    V = VOCAB
    net = SelfWiringGraph(n_neurons, V)
    td = BIGRAM_DIST
    ai = ACTIVE_INPUTS

    score = score_hybrid_bigram(net, td, V, ai, ticks)
    best_score = score
    kept = 0
    stale = 0
    temperature = 1.0
    phase = "STRUCTURE"

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if temp_mode == 'baseline':
            if phase == "STRUCTURE":
                capped_mutate_once(net, 0.05)
            else:
                if random.random() < 0.3:
                    capped_mutate_once(net, 0.02)
                else:
                    net.mutate_weights()
        else:
            temp_mutate(net, temperature)

        new_score = score_hybrid_bigram(net, td, V, ai, ticks)

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
            if temp_mode == 'temp_gradual':
                temperature = max(0.3, temperature * 0.95)
            elif temp_mode == 'temp_aggressive':
                temperature = max(0.2, temperature * 0.90)
        else:
            net.restore_state(state)
            stale += 1
            if temp_mode == 'temp_gradual':
                if stale % 200 == 0:
                    temperature = min(5.0, temperature * 1.3)
            elif temp_mode == 'temp_aggressive':
                if stale % 200 == 0:
                    temperature = min(8.0, temperature * 1.5)

        # Phase transition (baseline only)
        if temp_mode == 'baseline' and phase == "STRUCTURE" and stale > 2500:
            phase = "BOTH"
            stale = 0

        if stale >= 6000:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100
    final = eval_bigram_metrics(net, td, V, ai, ticks)

    return {
        'label': label,
        'seed': seed,
        'temp_mode': temp_mode,
        'final_mse': final['mse'],
        'final_top1': final['top1'],
        'final_top3': final['top3'],
        'accept_rate': accept_rate,
        'kept': kept,
        'attempts': att + 1,
        'conns': net.count_connections(),
        'time': elapsed,
        'final_temp': temperature,
    }


# ============================================================
#  Workers
# ============================================================

def worker_synth(args):
    return train_synthetic_temp(**args)

def worker_local(args):
    return train_synthetic_local(**args)

def worker_bigram(args):
    return train_bigram_temp(**args)


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]
TEMP_MODES = ['baseline', 'temp_gradual', 'temp_aggressive', 'temp_step']
LOCAL_MODES = ['global_temp', 'local_temp', 'hybrid']

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24

    print(f"  Temperature-Modulated Mutation Test — v22 SelfWiringGraph")
    print(f"  {'=' * 60}")
    print(f"  Temp modes: {TEMP_MODES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  CPU cores: {n_workers}")
    print(f"  Density cap: {MAX_DENSITY * 100:.0f}%")

    # =========================================================
    # PHASE A: Global temperature — synthetic tasks
    # =========================================================
    print(f"\n{'#' * 65}")
    print(f"  PHASE A: Global temperature (160n, 8K att, 5 seeds)")
    print(f"  4 temp modes x 3 tasks x 5 seeds = 60 jobs")
    print(f"{'#' * 65}", flush=True)

    configs_a = []
    for n_classes in [16, 32, 64]:
        for tm in TEMP_MODES:
            for seed in SEEDS:
                configs_a.append(dict(
                    label=f'{tm}_{n_classes}c',
                    n_neurons=160, seed=seed, temp_mode=tm,
                    n_classes=n_classes, max_attempts=8000, ticks=8))

    print(f"\n  Running {len(configs_a)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_a = list(pool.map(worker_synth, configs_a))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    # Aggregate Phase A
    groups_a = defaultdict(list)
    for r in results_a:
        groups_a[r['label']].append(r)

    for n_classes in [16, 32, 64]:
        print(f"\n  --- {n_classes}-class ---")
        print(f"  {'Temp Mode':<22s} {'Acc':>7s} {'Conns':>7s} "
              f"{'Accept%':>9s} {'FinalT':>7s} {'Time':>6s}")
        print(f"  {'-' * 62}")
        for tm in TEMP_MODES:
            label = f'{tm}_{n_classes}c'
            runs = groups_a.get(label, [])
            if not runs:
                continue
            accs = [r['final_acc'] for r in runs]
            conns = [r['conns'] for r in runs]
            ars = [r['accept_rate'] for r in runs]
            tms_time = [r['time'] for r in runs]
            ftemps = [r.get('final_temp', 0) for r in runs]
            print(f"  {tm:<22s} {np.mean(accs) * 100:5.1f}% {np.mean(conns):5.0f} "
                  f"{np.mean(ars):7.2f}% {np.mean(ftemps):5.2f} {np.mean(tms_time):5.0f}s")

    # Rank by geometric mean
    print(f"\n  --- OVERALL RANKING (geometric mean of accuracy) ---")
    tm_scores = {}
    for tm in TEMP_MODES:
        accs_all = []
        for n_classes in [16, 32, 64]:
            label = f'{tm}_{n_classes}c'
            runs = groups_a.get(label, [])
            if runs:
                accs_all.append(np.mean([r['final_acc'] for r in runs]))
        if accs_all:
            geo_mean = np.exp(np.mean(np.log(np.maximum(accs_all, 1e-6))))
            tm_scores[tm] = geo_mean

    for rank, (tm, gm) in enumerate(sorted(tm_scores.items(),
                                            key=lambda x: -x[1]), 1):
        marker = " <<<" if rank == 1 else ""
        print(f"  #{rank} {tm:<22s} geo_mean={gm * 100:.1f}%{marker}")

    # Temperature trajectory
    print(f"\n  --- TEMPERATURE TRAJECTORY (sampled every 1000 steps) ---")
    for tm in TEMP_MODES:
        if tm == 'baseline':
            continue
        label = f'{tm}_64c'
        runs = groups_a.get(label, [])
        if runs and runs[0].get('temp_log'):
            log = runs[0]['temp_log']
            traj = ' -> '.join([f'{t:.2f}' for t in log[:9]])
            print(f"  {tm:<22s} {traj}")

    # =========================================================
    # Check if Phase A is positive
    # =========================================================
    best_tm = max(tm_scores, key=tm_scores.get) if tm_scores else 'baseline'
    baseline_gm = tm_scores.get('baseline', 0)
    best_gm = tm_scores.get(best_tm, 0)
    phase_a_positive = best_tm != 'baseline' and best_gm > baseline_gm * 1.02

    print(f"\n  Phase A verdict: best={best_tm} ({best_gm * 100:.1f}%) "
          f"vs baseline ({baseline_gm * 100:.1f}%)")
    if phase_a_positive:
        print(f"  >>> POSITIVE (+{(best_gm - baseline_gm) * 100:.1f}%) -> "
              f"proceeding to Phase B")
    else:
        # Still run Phase B for completeness
        print(f"  >>> MARGINAL or NEGATIVE -> running Phase B anyway for data")

    # =========================================================
    # PHASE B: Local temperature (64-class only)
    # =========================================================
    print(f"\n{'#' * 65}")
    print(f"  PHASE B: Local temperature (160n, 8K att, 64-class, 5 seeds)")
    print(f"  3 local modes x 5 seeds = 15 jobs")
    print(f"{'#' * 65}", flush=True)

    configs_b = []
    for lm in LOCAL_MODES:
        for seed in SEEDS:
            configs_b.append(dict(
                label=f'{lm}_64c',
                n_neurons=160, seed=seed, local_mode=lm,
                n_classes=64, max_attempts=8000, ticks=8))

    print(f"\n  Running {len(configs_b)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_b = list(pool.map(worker_local, configs_b))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    # Aggregate Phase B
    groups_b = defaultdict(list)
    for r in results_b:
        groups_b[r['label']].append(r)

    print(f"\n  {'Local Mode':<22s} {'Acc':>7s} {'Conns':>7s} "
          f"{'Accept%':>9s} {'Time':>6s}")
    print(f"  {'-' * 55}")
    for lm in LOCAL_MODES:
        label = f'{lm}_64c'
        runs = groups_b.get(label, [])
        if not runs:
            continue
        accs = [r['final_acc'] for r in runs]
        conns = [r['conns'] for r in runs]
        ars = [r['accept_rate'] for r in runs]
        tms_time = [r['time'] for r in runs]
        print(f"  {lm:<22s} {np.mean(accs) * 100:5.1f}% {np.mean(conns):5.0f} "
              f"{np.mean(ars):7.2f}% {np.mean(tms_time):5.0f}s")

    # Compare with Phase A 64-class baseline
    baseline_64 = groups_a.get('baseline_64c', [])
    if baseline_64:
        bl_acc = np.mean([r['final_acc'] for r in baseline_64])
        print(f"\n  Phase A baseline (64c): {bl_acc * 100:.1f}%")

    # Phase B ranking
    lm_scores = {}
    for lm in LOCAL_MODES:
        label = f'{lm}_64c'
        runs = groups_b.get(label, [])
        if runs:
            lm_scores[lm] = np.mean([r['final_acc'] for r in runs])

    print(f"\n  --- LOCAL TEMP RANKING ---")
    for rank, (lm, acc) in enumerate(sorted(lm_scores.items(),
                                             key=lambda x: -x[1]), 1):
        marker = " <<<" if rank == 1 else ""
        print(f"  #{rank} {lm:<22s} acc={acc * 100:.1f}%{marker}")

    # =========================================================
    # PHASE C: English bigram
    # =========================================================
    # Pick best temp mode from Phase A (or gradual as default)
    bigram_modes = ['baseline']
    if best_tm != 'baseline':
        bigram_modes.append(best_tm)
    if 'temp_gradual' not in bigram_modes:
        bigram_modes.append('temp_gradual')

    print(f"\n{'#' * 65}")
    print(f"  PHASE C: English bigram (160n, 8K att, hybrid scoring)")
    print(f"  {len(bigram_modes)} modes x 5 seeds = {len(bigram_modes) * 5} jobs")
    print(f"{'#' * 65}", flush=True)

    configs_c = []
    for tm in bigram_modes:
        for seed in SEEDS:
            configs_c.append(dict(
                label=f'{tm}_bigram',
                n_neurons=160, seed=seed, temp_mode=tm,
                max_attempts=8000, ticks=8))

    print(f"\n  Running {len(configs_c)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_c = list(pool.map(worker_bigram, configs_c))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    # Aggregate Phase C
    groups_c = defaultdict(list)
    for r in results_c:
        groups_c[r['label']].append(r)

    print(f"\n  {'Temp Mode':<22s} {'MSE':>10s} {'Top-1':>7s} {'Top-3':>7s} "
          f"{'Conns':>7s} {'Accept%':>9s} {'FinalT':>7s} {'Time':>6s}")
    print(f"  {'-' * 80}")
    for tm in bigram_modes:
        label = f'{tm}_bigram'
        runs = groups_c.get(label, [])
        if not runs:
            continue
        mses = [r['final_mse'] for r in runs]
        t1s = [r['final_top1'] for r in runs]
        t3s = [r['final_top3'] for r in runs]
        conns = [r['conns'] for r in runs]
        ars = [r['accept_rate'] for r in runs]
        ftemps = [r.get('final_temp', 0) for r in runs]
        tms_time = [r['time'] for r in runs]
        print(f"  {tm:<22s} {np.mean(mses):.6f} {np.mean(t1s) * 100:5.1f}% "
              f"{np.mean(t3s) * 100:5.1f}% {np.mean(conns):5.0f} "
              f"{np.mean(ars):7.2f}% {np.mean(ftemps):5.2f} {np.mean(tms_time):5.0f}s")

    # =========================================================
    # FINAL VERDICT
    # =========================================================
    print(f"\n  {'=' * 65}")
    print(f"  FINAL VERDICT")
    print(f"  {'=' * 65}")

    print(f"\n  Phase A (synthetic):")
    for rank, (tm, gm) in enumerate(sorted(tm_scores.items(),
                                            key=lambda x: -x[1]), 1):
        marker = " <<<" if rank == 1 else ""
        print(f"    #{rank} {tm:<22s} geo_mean={gm * 100:.1f}%{marker}")

    print(f"\n  Phase B (local temp, 64-class):")
    for rank, (lm, acc) in enumerate(sorted(lm_scores.items(),
                                             key=lambda x: -x[1]), 1):
        marker = " <<<" if rank == 1 else ""
        print(f"    #{rank} {lm:<22s} acc={acc * 100:.1f}%{marker}")

    print(f"\n  Phase C (bigram):")
    bigram_ranking = {}
    for tm in bigram_modes:
        label = f'{tm}_bigram'
        runs = groups_c.get(label, [])
        if runs:
            bigram_ranking[tm] = np.mean([r['final_mse'] for r in runs])
    for rank, (tm, mse) in enumerate(sorted(bigram_ranking.items(),
                                             key=lambda x: x[1]), 1):
        top1 = np.mean([r['final_top1']
                         for r in groups_c[f'{tm}_bigram']]) * 100
        marker = " <<<" if rank == 1 else ""
        print(f"    #{rank} {tm:<22s} MSE={mse:.6f} Top1={top1:.1f}%{marker}")

    print(f"\n  {'=' * 65}")
    print(f"  DONE")
    print(f"  {'=' * 65}", flush=True)
