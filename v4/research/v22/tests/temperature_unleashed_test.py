"""
Temperature UNLEASHED Test -- v22 SelfWiringGraph
===================================================
All shackles removed: NO density cap, wider temp range (0.1-10),
16K attempts, faster heating (every 100 stale).

Previous test showed temperature stuck on floor (88.8% in FOCUSED zone).
This test lets it breathe.

Phase A: 4 modes x 3 tasks x 5 seeds = 60 jobs (16K att, NO CAP)
Phase B: 64-class DEEP RUN (32K att, 3 seeds)
Phase C: English bigram (16K att)
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
#  English text corpus
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
#  Mutation -- NO DENSITY CAP
# ============================================================

def flip_single(net):
    alive = np.argwhere(net.mask != 0)
    if len(alive) == 0:
        return
    idx = alive[random.randint(0, len(alive) - 1)]
    net.mask[int(idx[0]), int(idx[1])] *= -1


def temp_mutate_unleashed(net, temperature):
    """Temperature-modulated mutation WITHOUT density cap."""
    r = random.random()

    if temperature < 0.3:
        # ULTRA-FOCUSED: single flip only
        flip_single(net)

    elif temperature < 1.0:
        # FOCUSED: flip-heavy
        if r < 0.5:
            flip_single(net)
        else:
            net.mutate_structure(0.03)

    elif temperature < 2.0:
        # NORMAL: standard mixed
        if r < 0.3:
            flip_single(net)
        else:
            net.mutate_structure(0.05)

    elif temperature < 5.0:
        # WIDE: block mutation
        n_changes = int(2 + temperature)
        for _ in range(n_changes):
            if random.random() < 0.4:
                flip_single(net)
            else:
                net.mutate_structure(0.05)

    else:
        # EARTHQUAKE: massive rewire + region flip
        n_changes = int(temperature * 2)
        for _ in range(n_changes):
            net.mutate_structure(0.05)
        # Region flip
        region_size = min(int(temperature * 2), net.N)
        region = random.sample(range(net.N), region_size)
        for n in region:
            alive = np.argwhere(net.mask[n] != 0).flatten()
            if len(alive) > 0:
                idx = alive[random.randint(0, len(alive) - 1)]
                net.mask[n, idx] *= -1

    # Weight mutation: temperature-scaled (30% chance)
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

def score_combined_synthetic(net, targets, vocab, ticks=8):
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


def score_combined_perclass(net, targets, vocab, ticks=8):
    net.reset()
    correct = 0
    total_score = 0.0
    per_class = np.zeros(vocab, dtype=np.float32)
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
                per_class[inp] = acc_i
    acc = correct / vocab
    net.last_acc = acc
    return total_score / vocab, acc, per_class


def score_hybrid_bigram(net, true_dist, vocab, active_inputs, ticks=8):
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
    return 0.5 * top1_acc + 0.5 * (1.0 - mean_mse / 0.01)


def eval_bigram_metrics(net, true_dist, vocab, active_inputs, ticks=8):
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
#  Training -- Phase A: global temperature modes
# ============================================================

def train_synthetic(label, n_neurons, seed, temp_mode, n_classes,
                    max_attempts=16000, ticks=8):
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
    conns_log = []

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if temp_mode == 'baseline':
            if phase == "STRUCTURE":
                net.mutate_structure(0.05)
            else:
                if random.random() < 0.3:
                    net.mutate_structure(0.02)
                else:
                    net.mutate_weights()

        elif temp_mode == 'local_temp':
            # Use per-class targeting with global temp
            temp_mutate_unleashed(net, temperature)

        else:
            temp_mutate_unleashed(net, temperature)

        new_score, new_acc = score_combined_synthetic(net, perm, V, ticks)

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)

            if temp_mode == 'temp_gradual':
                temperature = max(0.1, temperature * 0.95)
            elif temp_mode == 'temp_wide':
                temperature = max(0.1, temperature * 0.90)
            elif temp_mode == 'local_temp':
                temperature = max(0.1, temperature * 0.95)
        else:
            net.restore_state(state)
            stale += 1

            if temp_mode == 'temp_gradual':
                if stale % 100 == 0:
                    temperature = min(10.0, temperature * 1.3)
            elif temp_mode == 'temp_wide':
                if stale % 50 == 0:
                    temperature = min(15.0, temperature * 1.4)
            elif temp_mode == 'local_temp':
                if stale % 100 == 0:
                    temperature = min(10.0, temperature * 1.3)

        # Phase transition (baseline only)
        if temp_mode == 'baseline' and phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0

        if att % 2000 == 0:
            temp_log.append(temperature)
            conns_log.append(net.count_connections())

        if best_acc >= 0.99:
            break
        if temp_mode == 'baseline' and stale >= 8000:
            break
        if temp_mode != 'baseline' and stale >= 8000:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100

    return {
        'label': label, 'seed': seed, 'temp_mode': temp_mode,
        'n_classes': n_classes, 'final_acc': best_acc,
        'accept_rate': accept_rate, 'kept': kept,
        'attempts': att + 1, 'conns': net.count_connections(),
        'time': elapsed, 'final_temp': temperature,
        'temp_log': temp_log, 'conns_log': conns_log,
    }


# ============================================================
#  Training -- Phase B: deep run
# ============================================================

def train_deep(label, n_neurons, seed, temp_mode, n_classes,
               max_attempts=32000, ticks=8):
    """Same as train_synthetic but 32K attempts."""
    return train_synthetic(label, n_neurons, seed, temp_mode, n_classes,
                           max_attempts=max_attempts, ticks=ticks)


# ============================================================
#  Training -- Phase C: bigram
# ============================================================

def train_bigram(label, n_neurons, seed, temp_mode,
                 max_attempts=16000, ticks=8):
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
                net.mutate_structure(0.05)
            else:
                if random.random() < 0.3:
                    net.mutate_structure(0.02)
                else:
                    net.mutate_weights()
        else:
            temp_mutate_unleashed(net, temperature)

        new_score = score_hybrid_bigram(net, td, V, ai, ticks)

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
            if temp_mode != 'baseline':
                temperature = max(0.1, temperature * 0.95)
        else:
            net.restore_state(state)
            stale += 1
            if temp_mode != 'baseline' and stale % 100 == 0:
                temperature = min(10.0, temperature * 1.3)

        if temp_mode == 'baseline' and phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0

        if stale >= 8000:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100
    final = eval_bigram_metrics(net, td, V, ai, ticks)

    return {
        'label': label, 'seed': seed, 'temp_mode': temp_mode,
        'final_mse': final['mse'], 'final_top1': final['top1'],
        'final_top3': final['top3'], 'accept_rate': accept_rate,
        'kept': kept, 'attempts': att + 1,
        'conns': net.count_connections(), 'time': elapsed,
        'final_temp': temperature,
    }


# ============================================================
#  Workers
# ============================================================

def worker_synth(args):
    return train_synthetic(**args)

def worker_deep(args):
    return train_deep(**args)

def worker_bigram(args):
    return train_bigram(**args)


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]
TEMP_MODES = ['baseline', 'temp_gradual', 'temp_wide', 'local_temp']

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24

    print(f"  Temperature UNLEASHED Test -- v22 SelfWiringGraph")
    print(f"  {'=' * 60}")
    print(f"  Modes: {TEMP_MODES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  CPU cores: {n_workers}")
    print(f"  DENSITY CAP: OFF")
    print(f"  Temp range: 0.1 - 10.0 (gradual), 0.1 - 15.0 (wide)")
    print(f"  Max attempts: 16K (Phase A), 32K (Phase B)")

    # =========================================================
    # PHASE A: Global temperature UNLEASHED
    # =========================================================
    print(f"\n{'#' * 65}")
    print(f"  PHASE A: Temperature UNLEASHED (160n, 16K att, NO CAP)")
    print(f"  4 modes x 3 tasks x 5 seeds = 60 jobs")
    print(f"{'#' * 65}", flush=True)

    configs_a = []
    for n_classes in [16, 32, 64]:
        for tm in TEMP_MODES:
            for seed in SEEDS:
                configs_a.append(dict(
                    label=f'{tm}_{n_classes}c',
                    n_neurons=160, seed=seed, temp_mode=tm,
                    n_classes=n_classes, max_attempts=16000, ticks=8))

    print(f"\n  Running {len(configs_a)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_a = list(pool.map(worker_synth, configs_a))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    # Aggregate
    groups_a = defaultdict(list)
    for r in results_a:
        groups_a[r['label']].append(r)

    for n_classes in [16, 32, 64]:
        print(f"\n  --- {n_classes}-class ---")
        print(f"  {'Mode':<18s} {'Acc':>7s} {'Conns':>7s} "
              f"{'Accept%':>9s} {'FinalT':>7s} {'Time':>6s}")
        print(f"  {'-' * 58}")
        for tm in TEMP_MODES:
            label = f'{tm}_{n_classes}c'
            runs = groups_a.get(label, [])
            if not runs:
                continue
            accs = [r['final_acc'] for r in runs]
            conns = [r['conns'] for r in runs]
            ars = [r['accept_rate'] for r in runs]
            tms = [r['time'] for r in runs]
            ftemps = [r.get('final_temp', 0) for r in runs]
            print(f"  {tm:<18s} {np.mean(accs)*100:5.1f}% {np.mean(conns):5.0f} "
                  f"{np.mean(ars):7.2f}% {np.mean(ftemps):5.2f} {np.mean(tms):5.0f}s")

    # Ranking
    print(f"\n  --- OVERALL RANKING (geometric mean) ---")
    tm_scores = {}
    for tm in TEMP_MODES:
        accs_all = []
        for n_classes in [16, 32, 64]:
            runs = groups_a.get(f'{tm}_{n_classes}c', [])
            if runs:
                accs_all.append(np.mean([r['final_acc'] for r in runs]))
        if accs_all:
            tm_scores[tm] = np.exp(np.mean(np.log(np.maximum(accs_all, 1e-6))))

    for rank, (tm, gm) in enumerate(sorted(tm_scores.items(),
                                            key=lambda x: -x[1]), 1):
        marker = " <<<" if rank == 1 else ""
        print(f"  #{rank} {tm:<18s} geo_mean={gm*100:.1f}%{marker}")

    # Temperature trajectory (64-class, first seed)
    print(f"\n  --- TEMP TRAJECTORY (64-class, seed=42, every 2K steps) ---")
    for tm in TEMP_MODES:
        if tm == 'baseline':
            continue
        runs = groups_a.get(f'{tm}_64c', [])
        if runs:
            r0 = runs[0]
            if r0.get('temp_log'):
                traj = ' -> '.join([f'{t:.2f}' for t in r0['temp_log'][:9]])
                print(f"  {tm:<18s} T: {traj}")
            if r0.get('conns_log'):
                ctraj = ' -> '.join([f'{c}' for c in r0['conns_log'][:9]])
                print(f"  {'':<18s} C: {ctraj}")

    # Connection count comparison
    print(f"\n  --- CONNECTION COUNTS (64-class, mean over seeds) ---")
    for tm in TEMP_MODES:
        runs = groups_a.get(f'{tm}_64c', [])
        if runs:
            conns = [r['conns'] for r in runs]
            density = np.mean(conns) / (160 * 159) * 100
            print(f"  {tm:<18s} {np.mean(conns):.0f} conns ({density:.1f}%)")

    best_tm = max(tm_scores, key=tm_scores.get) if tm_scores else 'baseline'
    baseline_gm = tm_scores.get('baseline', 0)
    best_gm = tm_scores.get(best_tm, 0)

    print(f"\n  Phase A verdict: {best_tm} ({best_gm*100:.1f}%) "
          f"vs baseline ({baseline_gm*100:.1f}%)")

    # =========================================================
    # PHASE B: 64-class DEEP RUN (32K attempts)
    # =========================================================
    deep_modes = [best_tm] if best_tm != 'baseline' else ['temp_gradual']
    deep_modes.append('baseline')
    deep_seeds = [42, 123, 777]

    print(f"\n{'#' * 65}")
    print(f"  PHASE B: 64-class DEEP RUN (160n, 32K att, NO CAP)")
    print(f"  {len(deep_modes)} modes x {len(deep_seeds)} seeds = "
          f"{len(deep_modes)*len(deep_seeds)} jobs")
    print(f"{'#' * 65}", flush=True)

    configs_b = []
    for tm in deep_modes:
        for seed in deep_seeds:
            configs_b.append(dict(
                label=f'{tm}_deep64',
                n_neurons=160, seed=seed, temp_mode=tm,
                n_classes=64, max_attempts=32000, ticks=8))

    print(f"\n  Running {len(configs_b)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_b = list(pool.map(worker_deep, configs_b))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    groups_b = defaultdict(list)
    for r in results_b:
        groups_b[r['label']].append(r)

    print(f"\n  {'Mode':<18s} {'Acc':>7s} {'Conns':>7s} "
          f"{'Accept%':>9s} {'Steps':>7s} {'Time':>6s}")
    print(f"  {'-' * 58}")
    for tm in deep_modes:
        label = f'{tm}_deep64'
        runs = groups_b.get(label, [])
        if not runs:
            continue
        accs = [r['final_acc'] for r in runs]
        conns = [r['conns'] for r in runs]
        ars = [r['accept_rate'] for r in runs]
        steps = [r['attempts'] for r in runs]
        tms = [r['time'] for r in runs]
        print(f"  {tm:<18s} {np.mean(accs)*100:5.1f}% {np.mean(conns):5.0f} "
              f"{np.mean(ars):7.2f}% {np.mean(steps):6.0f} {np.mean(tms):5.0f}s")

    # =========================================================
    # PHASE C: English bigram
    # =========================================================
    bigram_modes = ['baseline']
    if best_tm != 'baseline':
        bigram_modes.append(best_tm)
    if 'temp_gradual' not in bigram_modes:
        bigram_modes.append('temp_gradual')

    print(f"\n{'#' * 65}")
    print(f"  PHASE C: English bigram UNLEASHED (160n, 16K att, NO CAP)")
    print(f"  {len(bigram_modes)} modes x 5 seeds = {len(bigram_modes)*5} jobs")
    print(f"{'#' * 65}", flush=True)

    configs_c = []
    for tm in bigram_modes:
        for seed in SEEDS:
            configs_c.append(dict(
                label=f'{tm}_bigram',
                n_neurons=160, seed=seed, temp_mode=tm,
                max_attempts=16000, ticks=8))

    print(f"\n  Running {len(configs_c)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_c = list(pool.map(worker_bigram, configs_c))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    groups_c = defaultdict(list)
    for r in results_c:
        groups_c[r['label']].append(r)

    print(f"\n  {'Mode':<18s} {'MSE':>10s} {'Top-1':>7s} {'Top-3':>7s} "
          f"{'Conns':>7s} {'Accept%':>9s} {'Time':>6s}")
    print(f"  {'-' * 70}")
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
        tms = [r['time'] for r in runs]
        print(f"  {tm:<18s} {np.mean(mses):.6f} {np.mean(t1s)*100:5.1f}% "
              f"{np.mean(t3s)*100:5.1f}% {np.mean(conns):5.0f} "
              f"{np.mean(ars):7.2f}% {np.mean(tms):5.0f}s")

    # =========================================================
    # FINAL VERDICT
    # =========================================================
    print(f"\n  {'=' * 65}")
    print(f"  FINAL VERDICT")
    print(f"  {'=' * 65}")

    print(f"\n  Phase A (synthetic, 16K att, NO CAP):")
    for rank, (tm, gm) in enumerate(sorted(tm_scores.items(),
                                            key=lambda x: -x[1]), 1):
        marker = " <<<" if rank == 1 else ""
        print(f"    #{rank} {tm:<18s} geo_mean={gm*100:.1f}%{marker}")

    print(f"\n  Phase B (64-class deep 32K):")
    for tm in deep_modes:
        runs = groups_b.get(f'{tm}_deep64', [])
        if runs:
            acc = np.mean([r['final_acc'] for r in runs])
            conns = np.mean([r['conns'] for r in runs])
            print(f"    {tm:<18s} acc={acc*100:.1f}% conns={conns:.0f}")

    print(f"\n  Phase C (bigram, 16K):")
    bigram_ranking = {}
    for tm in bigram_modes:
        runs = groups_c.get(f'{tm}_bigram', [])
        if runs:
            bigram_ranking[tm] = np.mean([r['final_mse'] for r in runs])
    for rank, (tm, mse) in enumerate(sorted(bigram_ranking.items(),
                                             key=lambda x: x[1]), 1):
        top1 = np.mean([r['final_top1']
                         for r in groups_c[f'{tm}_bigram']]) * 100
        marker = " <<<" if rank == 1 else ""
        print(f"    #{rank} {tm:<18s} MSE={mse:.6f} Top1={top1:.1f}%{marker}")

    # ATR comparison
    print(f"\n  64-class ALL-TIME RECORD: 64.4% (16K att, no temp)")
    best_64 = 0
    for tm in TEMP_MODES:
        runs = groups_a.get(f'{tm}_64c', [])
        if runs:
            best_64 = max(best_64, max(r['final_acc'] for r in runs))
    runs_deep = []
    for tm in deep_modes:
        runs_deep.extend(groups_b.get(f'{tm}_deep64', []))
    if runs_deep:
        best_64 = max(best_64, max(r['final_acc'] for r in runs_deep))
    print(f"  Best 64-class this test: {best_64*100:.1f}%")
    if best_64 > 0.644:
        print(f"  >>> NEW ALL-TIME RECORD! <<<")

    print(f"\n  {'=' * 65}")
    print(f"  DONE")
    print(f"  {'=' * 65}", flush=True)
