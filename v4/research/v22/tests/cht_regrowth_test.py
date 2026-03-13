"""
CHT Regrowth Test — v22 SelfWiringGraph (v2, density-capped)
==============================================================
Cannistraci-Hebb Training (NeurIPS 2025) style topology-driven
self-wiring vs no wiring.

Core idea: if A→B and B→C exist, then A→C is likely needed.
score(u,v) = number of common neighbors = (A @ A)[u,v]

At N=160, A@A is a 160×160 matmul — microseconds. Zero overhead.

v2 fixes from sanity check:
  - Density cap (MAX_DENSITY=0.15): mutate_structure add ops are skipped
    if density exceeds cap. Without this, all modes converge to ~25K conns
    (near-full graph) drowning out wiring signal.
  - 4d_address mode removed: self_wire() never fires because internal
    neuron state is always zero when called (before forward pass).
    This is a bug in the base class, not worth working around here.

Phase A: Synthetic tasks (16/32/64-class) — combined scoring
Phase B: English bigram — hybrid scoring
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
idx_to_char = {i: c for c, i in char_to_idx.items()}
VOCAB = len(chars)

def compute_bigram_dist():
    counts = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for i in range(len(TEXT) - 1):
        a, b = TEXT[i], TEXT[i+1]
        if a in char_to_idx and b in char_to_idx:
            counts[char_to_idx[a], char_to_idx[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums

BIGRAM_DIST = compute_bigram_dist()
ACTIVE_INPUTS = [i for i in range(VOCAB) if BIGRAM_DIST[i].sum() > 0.01]


# ============================================================
#  CHT wiring functions
# ============================================================

def cht_wire(net, max_new=3):
    """CHT grow-only: add connections with highest common-neighbor score."""
    A = (net.mask != 0).astype(np.float32)
    np.fill_diagonal(A, 0)
    C = A @ A  # common neighbor matrix
    C[net.mask != 0] = 0  # exclude existing connections
    np.fill_diagonal(C, 0)  # no self-loops

    cmax = C.max()
    if cmax == 0:
        return 0

    flat = C.ravel()
    n_pos = int((flat > 0).sum())
    if n_pos == 0:
        return 0

    k = min(max_new * 3, n_pos)
    top_idx = np.argpartition(flat, -k)[-k:]
    top_idx = top_idx[flat[top_idx] > 0]
    top_idx = top_idx[np.argsort(-flat[top_idx])][:max_new]

    added = 0
    for idx in top_idx:
        r, c = divmod(int(idx), net.N)
        net.mask[r, c] = random.choice([-1.0, 1.0])
        net.W[r, c] = random.choice([np.float32(0.5), np.float32(1.5)])
        added += 1
    return added


def cht_prune_regrow(net, zeta=0.10):
    """CHT prune+regrow: remove poorly-supported, add well-supported."""
    A = (net.mask != 0).astype(np.float32)
    np.fill_diagonal(A, 0)
    C = A @ A

    alive = np.argwhere(net.mask != 0)
    if len(alive) < 10:
        return 0, 0

    # Score alive connections (low = poorly supported by topology)
    alive_scores = np.array([C[r, c] for r, c in alive])
    n_remove = max(1, int(len(alive) * zeta))

    # Remove bottom zeta
    remove_idx = np.argsort(alive_scores)[:n_remove]
    for i in remove_idx:
        net.mask[int(alive[i][0]), int(alive[i][1])] = 0

    # Recompute after removal
    A2 = (net.mask != 0).astype(np.float32)
    np.fill_diagonal(A2, 0)
    C2 = A2 @ A2
    C2[net.mask != 0] = 0
    np.fill_diagonal(C2, 0)

    # Regrow top n_remove
    flat = C2.ravel()
    added = 0
    if flat.max() > 0:
        n_pos = int((flat > 0).sum())
        k = min(n_remove * 3, n_pos)
        if k > 0:
            top_idx = np.argpartition(flat, -k)[-k:]
            top_idx = top_idx[flat[top_idx] > 0]
            top_idx = top_idx[np.argsort(-flat[top_idx])][:n_remove]
            for idx in top_idx:
                r, c = divmod(int(idx), net.N)
                net.mask[r, c] = random.choice([-1.0, 1.0])
                net.W[r, c] = random.choice([np.float32(0.5), np.float32(1.5)])
                added += 1

    return len(remove_idx), added


def wire_dispatch(net, mode, step):
    """Route to appropriate wiring method."""
    if mode == 'none':
        pass
    elif mode == 'cht':
        cht_wire(net, max_new=3)
    elif mode == 'cht_prune_regrow':
        if step % 500 == 0 and step > 0:
            cht_prune_regrow(net, zeta=0.10)
        else:
            cht_wire(net, max_new=3)
    elif mode == 'cht_arousal':
        if net.last_acc < 0.3:
            max_new = 1
        elif net.last_acc < 0.7:
            max_new = 2
        else:
            max_new = 3
        cht_wire(net, max_new=max_new)


# ============================================================
#  Density-capped mutation
# ============================================================

MAX_DENSITY = 0.15  # 15% max — initial is 6%

def density_capped_mutate(net, rate, phase):
    """Mutate structure but skip add ops if density exceeds MAX_DENSITY."""
    N = net.N
    max_conns = N * (N - 1)  # exclude diagonal
    current = int((net.mask != 0).sum())
    density = current / max_conns

    if phase == "STRUCTURE":
        mut_rate = 0.05
    else:
        mut_rate = 0.02

    if density >= MAX_DENSITY:
        # Only allow flip, remove, rewire — NOT add
        r = random.random()
        if r < net.flip_rate:
            # Flip: toggle sign of existing connections
            alive = np.argwhere(net.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * mut_rate * 0.5))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    net.mask[r2, c] *= -1
        else:
            action = random.choice(['remove', 'rewire'])
            if action == 'remove':
                alive = np.argwhere(net.mask != 0)
                if len(alive) > 3:
                    n = max(1, int(len(alive) * mut_rate))
                    idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                    for j in range(len(idx)):
                        net.mask[int(idx[j][0]), int(idx[j][1])] = 0
            else:  # rewire
                alive = np.argwhere(net.mask != 0)
                if len(alive) > 0:
                    n = max(1, int(len(alive) * mut_rate))
                    idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                    for j in range(len(idx)):
                        r2, c = int(idx[j][0]), int(idx[j][1])
                        old_sign = net.mask[r2, c]
                        old_w = net.W[r2, c]
                        net.mask[r2, c] = 0
                        nc = random.randint(0, N - 1)
                        while nc == r2:
                            nc = random.randint(0, N - 1)
                        net.mask[r2, nc] = old_sign
                        net.W[r2, nc] = old_w
    else:
        # Normal mutation (below density cap)
        net.mutate_structure(mut_rate)


# ============================================================
#  Scoring functions
# ============================================================

def score_combined_synthetic(net, targets, vocab, ticks=8):
    """Combined scoring for synthetic tasks: 0.5*acc + 0.5*target_prob."""
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


def score_hybrid_bigram(net, true_dist, vocab, active_inputs, ticks=8):
    """Hybrid scoring for bigram: 0.5*top1_match + 0.5*(1 - MSE/0.01)."""
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
#  Training
# ============================================================

def train_synthetic(label, n_neurons, seed, wire_mode, n_classes,
                    max_attempts=8000, ticks=8):
    """Train on synthetic permutation task with configurable wiring."""
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
    phase = "STRUCTURE"
    switched = False

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            density_capped_mutate(net, 0.05, "STRUCTURE")
        else:
            if random.random() < 0.3:
                density_capped_mutate(net, 0.02, "BOTH")
            else:
                net.mutate_weights()

        wire_dispatch(net, wire_mode, att)
        new_score, new_acc = score_combined_synthetic(net, perm, V, ticks)

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if best_acc >= 0.99:
            break
        if stale >= 3500:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100

    return {
        'label': label,
        'seed': seed,
        'wire_mode': wire_mode,
        'n_classes': n_classes,
        'final_acc': best_acc,
        'accept_rate': accept_rate,
        'kept': kept,
        'attempts': att + 1,
        'conns': net.count_connections(),
        'time': elapsed,
    }


def train_bigram(label, n_neurons, seed, wire_mode,
                 max_attempts=8000, ticks=8):
    """Train on English bigram with hybrid scoring + configurable wiring."""
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
    phase = "STRUCTURE"
    switched = False

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            density_capped_mutate(net, 0.05, "STRUCTURE")
        else:
            if random.random() < 0.3:
                density_capped_mutate(net, 0.02, "BOTH")
            else:
                net.mutate_weights()

        wire_dispatch(net, wire_mode, att)
        new_score = score_hybrid_bigram(net, td, V, ai, ticks)

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if stale >= 3500:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100

    final = eval_bigram_metrics(net, td, V, ai, ticks)

    return {
        'label': label,
        'seed': seed,
        'wire_mode': wire_mode,
        'final_mse': final['mse'],
        'final_top1': final['top1'],
        'final_top3': final['top3'],
        'accept_rate': accept_rate,
        'kept': kept,
        'attempts': att + 1,
        'conns': net.count_connections(),
        'time': elapsed,
    }


def worker_synth(args):
    return train_synthetic(**args)

def worker_bigram(args):
    return train_bigram(**args)


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]
WIRE_MODES = ['cht', 'cht_prune_regrow', 'cht_arousal', 'none']

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24

    print(f"  CHT Regrowth Test — v22 SelfWiringGraph")
    print(f"  {'='*60}")
    print(f"  Wiring modes: {WIRE_MODES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  CPU cores: {n_workers}")

    # =========================================================
    # PHASE A: Synthetic tasks
    # =========================================================
    print(f"  Density cap: {MAX_DENSITY*100:.0f}%")

    print(f"\n{'#'*65}")
    print(f"  PHASE A: Synthetic tasks (160n, 8K att, 5 seeds)")
    print(f"  4 wire modes x 3 tasks x 5 seeds = 60 jobs")
    print(f"{'#'*65}", flush=True)

    configs_a = []
    for n_classes in [16, 32, 64]:
        for wm in WIRE_MODES:
            for seed in SEEDS:
                configs_a.append(dict(
                    label=f'{wm}_{n_classes}c',
                    n_neurons=160, seed=seed, wire_mode=wm,
                    n_classes=n_classes, max_attempts=8000, ticks=8))

    print(f"\n  Running {len(configs_a)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_a = list(pool.map(worker_synth, configs_a))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    # Aggregate Phase A
    groups_a = defaultdict(list)
    for r in results_a:
        groups_a[r['label']].append(r)

    for n_classes in [16, 32, 64]:
        print(f"\n  --- {n_classes}-class ---")
        print(f"  {'Wire Mode':<22s} {'Acc':>7s} {'Conns':>7s} {'Accept%':>9s} {'Time':>6s}")
        print(f"  {'-'*55}")
        for wm in WIRE_MODES:
            label = f'{wm}_{n_classes}c'
            runs = groups_a.get(label, [])
            if not runs:
                continue
            accs = [r['final_acc'] for r in runs]
            conns = [r['conns'] for r in runs]
            ars = [r['accept_rate'] for r in runs]
            tms = [r['time'] for r in runs]
            print(f"  {wm:<22s} {np.mean(accs)*100:5.1f}% {np.mean(conns):5.0f} "
                  f"{np.mean(ars):7.2f}% {np.mean(tms):5.0f}s")

    # Rank by geometric mean across tasks
    print(f"\n  --- OVERALL RANKING (geometric mean of accuracy) ---")
    wm_scores = {}
    for wm in WIRE_MODES:
        accs_all = []
        for n_classes in [16, 32, 64]:
            label = f'{wm}_{n_classes}c'
            runs = groups_a.get(label, [])
            if runs:
                accs_all.append(np.mean([r['final_acc'] for r in runs]))
        if accs_all:
            geo_mean = np.exp(np.mean(np.log(np.maximum(accs_all, 1e-6))))
            wm_scores[wm] = geo_mean

    for rank, (wm, gm) in enumerate(sorted(wm_scores.items(),
                                            key=lambda x: -x[1]), 1):
        marker = " <<<" if rank == 1 else ""
        print(f"  #{rank} {wm:<22s} geo_mean={gm*100:.1f}%{marker}")

    # =========================================================
    # PHASE B: English bigram
    # =========================================================
    print(f"\n{'#'*65}")
    print(f"  PHASE B: English bigram (160n, 8K att, hybrid scoring)")
    print(f"  4 wire modes x 5 seeds = 20 jobs")
    print(f"{'#'*65}", flush=True)

    configs_b = []
    for wm in WIRE_MODES:
        for seed in SEEDS:
            configs_b.append(dict(
                label=f'{wm}_bigram',
                n_neurons=160, seed=seed, wire_mode=wm,
                max_attempts=8000, ticks=8))

    print(f"\n  Running {len(configs_b)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_b = list(pool.map(worker_bigram, configs_b))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    # Aggregate Phase B
    groups_b = defaultdict(list)
    for r in results_b:
        groups_b[r['label']].append(r)

    print(f"\n  {'Wire Mode':<22s} {'MSE':>10s} {'Top-1':>7s} {'Top-3':>7s} "
          f"{'Conns':>7s} {'Accept%':>9s} {'Time':>6s}")
    print(f"  {'-'*75}")
    for wm in WIRE_MODES:
        label = f'{wm}_bigram'
        runs = groups_b.get(label, [])
        if not runs:
            continue
        mses = [r['final_mse'] for r in runs]
        t1s = [r['final_top1'] for r in runs]
        t3s = [r['final_top3'] for r in runs]
        conns = [r['conns'] for r in runs]
        ars = [r['accept_rate'] for r in runs]
        tms = [r['time'] for r in runs]
        print(f"  {wm:<22s} {np.mean(mses):.6f} {np.mean(t1s)*100:5.1f}% "
              f"{np.mean(t3s)*100:5.1f}% {np.mean(conns):5.0f} "
              f"{np.mean(ars):7.2f}% {np.mean(tms):5.0f}s")

    # Bigram ranking
    print(f"\n  --- BIGRAM RANKING (by MSE, lower = better) ---")
    bigram_scores = {}
    for wm in WIRE_MODES:
        label = f'{wm}_bigram'
        runs = groups_b.get(label, [])
        if runs:
            bigram_scores[wm] = np.mean([r['final_mse'] for r in runs])

    for rank, (wm, mse) in enumerate(sorted(bigram_scores.items(),
                                             key=lambda x: x[1]), 1):
        top1 = np.mean([r['final_top1'] for r in groups_b[f'{wm}_bigram']]) * 100
        marker = " <<<" if rank == 1 else ""
        print(f"  #{rank} {wm:<22s} MSE={mse:.6f} Top1={top1:.1f}%{marker}")

    # =========================================================
    # VERDICT
    # =========================================================
    print(f"\n  {'='*65}")
    print(f"  VERDICT")
    print(f"  {'='*65}")

    # Best synthetic
    if wm_scores:
        best_synth = max(wm_scores, key=wm_scores.get)
        print(f"  Synthetic winner:  {best_synth} (geo_mean={wm_scores[best_synth]*100:.1f}%)")

    # Best bigram
    if bigram_scores:
        best_bigram = min(bigram_scores, key=bigram_scores.get)
        best_bigram_t1 = np.mean([r['final_top1']
                                  for r in groups_b[f'{best_bigram}_bigram']]) * 100
        print(f"  Bigram winner:     {best_bigram} (MSE={bigram_scores[best_bigram]:.6f}, "
              f"Top1={best_bigram_t1:.1f}%)")

    # Connection count comparison
    init_conns = int(160 * 160 * 0.06)
    max_capped = int(160 * (160 - 1) * MAX_DENSITY)
    print(f"\n  Connection counts (init ~{init_conns}, cap ~{max_capped} @ {MAX_DENSITY*100:.0f}%):")
    for wm in WIRE_MODES:
        label = f'{wm}_bigram'
        runs = groups_b.get(label, [])
        if runs:
            mean_conns = np.mean([r['conns'] for r in runs])
            density = mean_conns / (160 * 159) * 100
            print(f"    {wm:<22s} final={mean_conns:.0f} ({density:.1f}%)")

    print(f"\n  {'='*65}")
    print(f"  DONE")
    print(f"  {'='*65}", flush=True)
