"""
Batch + TAA Shield Combo Test -- v22 SelfWiringGraph
=====================================================
Does batch forward (17x) + TAA shield (+6.3%) stack?

4 modes, TIME-EQUIVALENT (each gets same wall-clock budget):
1. seq_random   — sequential eval + random mutation (baseline)
2. seq_taa      — sequential eval + TAA protected mutation
3. batch_random — batch eval + random mutation
4. batch_taa    — batch eval + TAA protected mutation

64-class, N=192, NO CAP, 5 seeds, parallel on all cores.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from concurrent.futures import ProcessPoolExecutor
from v22_best_config import SelfWiringGraph, softmax

# ============================================================
#  Batch forward (from batch_forward_test.py)
# ============================================================

def score_batch_vectorized(net, targets, V, ticks=8):
    """Fully vectorized: batch forward + batch softmax + batch scoring."""
    N = net.N
    Weff = net.W * net.mask
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

    logits = charges[:, :V]  # (V, V)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = (preds == targets).mean()
    tp = probs[np.arange(V), targets].mean()
    score = 0.5 * (preds == targets).astype(float).mean() + 0.5 * tp
    return score, float(acc)


def get_batch_activations(net, V, ticks=8):
    """Run batch forward and return per-neuron activation magnitudes (V, N)."""
    N = net.N
    Weff = net.W * net.mask
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

    return np.abs(acts)  # (V, N)


# ============================================================
#  Sequential scoring (original)
# ============================================================

def score_sequential(net, targets, V, ticks=8):
    net.reset()
    correct = 0
    total_score = 0.0
    for p in range(2):
        for inp in range(V):
            world = np.zeros(V, dtype=np.float32)
            world[inp] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:V])
            if p == 1:
                tgt = targets[inp]
                acc_i = 1.0 if np.argmax(probs) == tgt else 0.0
                tp = float(probs[tgt])
                total_score += 0.5 * acc_i + 0.5 * tp
                if acc_i > 0:
                    correct += 1
    return total_score / V, correct / V


def get_seq_activation(net, V, ticks=8):
    """Run 1 random input, return activation vector."""
    net.reset()
    inp = random.randint(0, V - 1)
    w = np.zeros(V, dtype=np.float32)
    w[inp] = 1.0
    net.forward(w, ticks)
    return np.abs(net.state)  # (N,)


# ============================================================
#  TAA Shield
# ============================================================

def update_heat_seq(conn_heat, net, V, alpha=0.02, ticks=8):
    """Update connection heatmap from 1 random sequential forward."""
    act = get_seq_activation(net, V, ticks)  # (N,)
    outer = np.outer(act, act) * np.abs(net.mask)
    conn_heat *= (1 - alpha)
    conn_heat += outer * alpha


def update_heat_batch(conn_heat, net, V, alpha=0.02, ticks=8):
    """Update connection heatmap from batch forward (mean over all inputs)."""
    acts = get_batch_activations(net, V, ticks)  # (V, N)
    mean_act = acts.mean(axis=0)  # (N,)
    outer = np.outer(mean_act, mean_act) * np.abs(net.mask)
    conn_heat *= (1 - alpha)
    conn_heat += outer * alpha


# ============================================================
#  Mutation: random (baseline) vs TAA protected
# ============================================================

def mutate_random(net, rate=0.05):
    """Standard random mutation (no protection)."""
    net.mutate_structure(rate)


def mutate_taa(net, conn_heat, rate=0.05):
    """TAA-protected mutation: shield hot connections."""
    N = net.N
    r = random.random()

    if r < 0.3:
        # FLIP — only COLD connections
        active = np.argwhere(net.mask != 0)
        if len(active) > 0:
            heats = conn_heat[active[:, 0], active[:, 1]]
            med = np.median(heats)
            cold = active[heats <= med]
            if len(cold) > 0:
                n = max(1, int(len(cold) * rate * 0.5))
                idx = cold[np.random.choice(len(cold), min(n, len(cold)), replace=False)]
            else:
                n = max(1, int(len(active) * rate * 0.5))
                idx = active[np.random.choice(len(active), min(n, len(active)), replace=False)]
            for j in range(len(idx)):
                net.mask[int(idx[j][0]), int(idx[j][1])] *= -1
    else:
        a = random.choice([0, 1, 2, 3])
        if a < 2:
            # ADD — anywhere (growth is free)
            dead = np.argwhere(net.mask == 0)
            dead = dead[dead[:, 0] != dead[:, 1]]
            if len(dead) > 0:
                n = max(1, int(len(dead) * rate))
                idx = dead[np.random.choice(len(dead), min(n, len(dead)), replace=False)]
                sign = 1.0 if a == 0 else -1.0
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    net.mask[r2, c] = sign
                    net.W[r2, c] = random.choice([np.float32(0.5), np.float32(1.5)])

        elif a == 2:
            # REMOVE — only COLD bottom third
            active = np.argwhere(net.mask != 0)
            if len(active) > 3:
                heats = conn_heat[active[:, 0], active[:, 1]]
                cold_idx = np.argsort(heats)[:len(heats) // 3]
                if len(cold_idx) > 0:
                    n = max(1, int(len(cold_idx) * rate))
                    picks = cold_idx[np.random.choice(len(cold_idx), min(n, len(cold_idx)), replace=False)]
                    for p in picks:
                        i = active[p]
                        net.mask[int(i[0]), int(i[1])] = 0
        else:
            # REWIRE — move cold connection
            active = np.argwhere(net.mask != 0)
            if len(active) > 0:
                heats = conn_heat[active[:, 0], active[:, 1]]
                cold_idx = np.argsort(heats)[:len(heats) // 3]
                if len(cold_idx) > 0:
                    pick = cold_idx[random.randint(0, len(cold_idx) - 1)]
                    i = active[pick]
                else:
                    i = active[random.randint(0, len(active) - 1)]
                old_sign = net.mask[int(i[0]), int(i[1])]
                old_w = net.W[int(i[0]), int(i[1])]
                net.mask[int(i[0]), int(i[1])] = 0
                nc = random.randint(0, N - 1)
                while nc == int(i[0]):
                    nc = random.randint(0, N - 1)
                net.mask[int(i[0]), nc] = old_sign
                net.W[int(i[0]), nc] = old_w

    # Weight toggle
    if random.random() < 0.3:
        active = np.argwhere(net.mask != 0)
        if len(active) > 0:
            i = active[random.randint(0, len(active) - 1)]
            net.W[int(i[0]), int(i[1])] = (
                np.float32(1.5) if net.W[int(i[0]), int(i[1])] < 1.0
                else np.float32(0.5))


# ============================================================
#  Time-based training loop
# ============================================================

def train_timed(mode, V, internal, seed, time_budget=45, ticks=8):
    """Train for a fixed wall-clock time budget. Returns results dict."""
    np.random.seed(seed)
    random.seed(seed)

    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)
    conn_heat = np.zeros((N, N), dtype=np.float32)

    use_batch = mode in ('batch_random', 'batch_taa')
    use_taa = mode in ('seq_taa', 'batch_taa')

    if use_batch:
        score, acc = score_batch_vectorized(net, perm, V, ticks)
    else:
        score, acc = score_sequential(net, perm, V, ticks)

    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    att = 0

    t0 = time.perf_counter()
    deadline = t0 + time_budget

    while time.perf_counter() < deadline:
        att += 1
        state = net.save_state()

        # TAA heatmap update (every 10 attempts to save overhead)
        if use_taa and att % 10 == 0:
            if use_batch:
                update_heat_batch(conn_heat, net, V, 0.02, ticks)
            else:
                update_heat_seq(conn_heat, net, V, 0.02, ticks)

        # Mutate
        if use_taa:
            if phase == "STRUCTURE":
                mutate_taa(net, conn_heat, 0.05)
            else:
                if random.random() < 0.3:
                    mutate_taa(net, conn_heat, 0.02)
                else:
                    net.mutate_weights()
        else:
            if phase == "STRUCTURE":
                mutate_random(net, 0.05)
            else:
                if random.random() < 0.3:
                    mutate_random(net, 0.02)
                else:
                    net.mutate_weights()

        # Evaluate
        if use_batch:
            new_score, new_acc = score_batch_vectorized(net, perm, V, ticks)
        else:
            new_score, new_acc = score_sequential(net, perm, V, ticks)

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

    elapsed = time.perf_counter() - t0
    accept_rate = (kept / max(att, 1)) * 100

    return {
        'mode': mode, 'seed': seed, 'V': V, 'N': N,
        'final_acc': best_acc, 'attempts': att, 'kept': kept,
        'accept_rate': accept_rate, 'conns': net.count_connections(),
        'time': elapsed,
    }


# ============================================================
#  Worker
# ============================================================

def worker(args):
    try:
        return train_timed(**args)
    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc(), **args}


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]
MODES = ['seq_random', 'seq_taa', 'batch_random', 'batch_taa']
TIME_BUDGET = 45  # seconds per job

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24

    print(f"  Batch + TAA Shield Combo Test")
    print(f"  {'='*55}")
    print(f"  Modes: {MODES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Time budget: {TIME_BUDGET}s per job")
    print(f"  CPU cores: {n_workers}")
    print(f"  Task: 64-class, N=192, NO CAP")

    V, internal = 64, 128  # N = V + internal = 192

    configs = []
    for mode in MODES:
        for seed in SEEDS:
            configs.append(dict(
                mode=mode, V=V, internal=internal,
                seed=seed, time_budget=TIME_BUDGET, ticks=8))

    print(f"\n  Running {len(configs)} jobs ({TIME_BUDGET}s each)...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker, configs))
    print(f"  Completed in {time.time() - t0:.0f}s", flush=True)

    # Check errors
    for r in results:
        if 'error' in r:
            print(f"  ERROR: {r['mode']} seed={r.get('seed')}: {r['error']}")

    # Aggregate
    print(f"\n  {'Mode':<16s} {'Acc':>7s} {'Attempts':>10s} {'Kept':>6s} "
          f"{'Accept%':>8s} {'Conns':>7s} {'Time':>6s}")
    print(f"  {'-'*65}")

    mode_accs = {}
    for mode in MODES:
        runs = [r for r in results if r.get('mode') == mode and 'error' not in r]
        if not runs:
            continue
        accs = [r['final_acc'] for r in runs]
        atts = [r['attempts'] for r in runs]
        kepts = [r['kept'] for r in runs]
        ars = [r['accept_rate'] for r in runs]
        conns = [r['conns'] for r in runs]
        tms = [r['time'] for r in runs]
        mode_accs[mode] = np.mean(accs)

        print(f"  {mode:<16s} {np.mean(accs)*100:5.1f}% {np.mean(atts):9.0f} "
              f"{np.mean(kepts):5.0f} {np.mean(ars):6.2f}% "
              f"{np.mean(conns):6.0f} {np.mean(tms):4.0f}s")

    # Per-seed detail
    print(f"\n  Per-seed accuracy:")
    print(f"  {'Seed':>6s}", end="")
    for mode in MODES:
        print(f"  {mode:>14s}", end="")
    print()
    for seed in SEEDS:
        print(f"  {seed:6d}", end="")
        for mode in MODES:
            r = [x for x in results if x.get('mode') == mode
                 and x.get('seed') == seed and 'error' not in x]
            if r:
                print(f"  {r[0]['final_acc']*100:12.1f}%", end="")
            else:
                print(f"  {'ERR':>14s}", end="")
        print()

    # Verdict
    print(f"\n  {'='*55}")
    print(f"  VERDICT")
    print(f"  {'='*55}")

    baseline = mode_accs.get('seq_random', 0)
    for mode in MODES:
        a = mode_accs.get(mode, 0)
        diff = (a - baseline) * 100
        sign = "+" if diff >= 0 else ""
        atts_mean = np.mean([r['attempts'] for r in results
                             if r.get('mode') == mode and 'error' not in r])
        print(f"  {mode:<16s}: {a*100:5.1f}% ({sign}{diff:.1f}%) "
              f"~{atts_mean:.0f} attempts in {TIME_BUDGET}s")

    # Stack check
    seq_taa_gain = (mode_accs.get('seq_taa', 0) - baseline) * 100
    batch_gain = (mode_accs.get('batch_random', 0) - baseline) * 100
    combo_gain = (mode_accs.get('batch_taa', 0) - baseline) * 100
    expected_sum = seq_taa_gain + batch_gain

    print(f"\n  TAA alone:   {seq_taa_gain:+.1f}%")
    print(f"  Batch alone: {batch_gain:+.1f}%")
    print(f"  Sum expected:{expected_sum:+.1f}%")
    print(f"  Combo actual:{combo_gain:+.1f}%")

    if combo_gain > expected_sum * 0.8:
        print(f"  -> STACKS! ({combo_gain:.1f}% >= {expected_sum*0.8:.1f}%)")
    elif combo_gain > max(seq_taa_gain, batch_gain):
        print(f"  -> PARTIAL STACK (better than either alone)")
    else:
        print(f"  -> DOES NOT STACK (one dominates)")

    print(f"\n  {'='*55}")
    print(f"  DONE")
    print(f"  {'='*55}", flush=True)
