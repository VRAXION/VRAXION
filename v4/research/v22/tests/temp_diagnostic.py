"""
Temperature Diagnostic — detailed trajectory analysis
======================================================
Single seed, all 3 temp modes, 64-class. Logs EVERY step.
Outputs: histogram, convergence zones, time-in-zone, extremes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax

MAX_DENSITY = 0.15


def flip_single(net):
    alive = np.argwhere(net.mask != 0)
    if len(alive) == 0:
        return
    idx = alive[random.randint(0, len(alive) - 1)]
    net.mask[int(idx[0]), int(idx[1])] *= -1


def capped_mutate_once(net, rate=0.03):
    N = net.N
    density = (net.mask != 0).sum() / (N * (N - 1))
    if density >= MAX_DENSITY:
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


def temp_mutate(net, temperature):
    if temperature < 0.5:
        flip_single(net)
    elif temperature < 1.5:
        r = random.random()
        if r < 0.3:
            flip_single(net)
        else:
            capped_mutate_once(net, 0.05)
    elif temperature < 3.0:
        n_changes = int(2 + temperature)
        for _ in range(n_changes):
            if random.random() < 0.5:
                flip_single(net)
            else:
                capped_mutate_once(net, 0.03)
    else:
        n_changes = int(temperature * 2)
        for _ in range(n_changes):
            capped_mutate_once(net, 0.03)
        N = net.N
        region = random.sample(range(N), min(10, N))
        for n in region:
            alive = np.argwhere(net.mask[n] != 0).flatten()
            if len(alive) > 0:
                idx = alive[random.randint(0, len(alive) - 1)]
                net.mask[n, idx] *= -1


def score_combined(net, targets, V, ticks=8):
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


def run_diagnostic(temp_mode, seed=42, n_classes=64, max_attempts=8000):
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    net = SelfWiringGraph(160, V)
    perm = np.random.permutation(V)

    score, acc = score_combined(net, perm, V)
    best_acc = acc
    kept = 0
    stale = 0
    temperature = 1.0

    # Detailed logs
    temp_history = []
    acc_history = []
    event_log = []  # (step, event, temp, acc)
    zone_time = {'FOCUSED': 0, 'NORMAL': 0, 'WIDE': 0, 'EARTHQUAKE': 0}

    for att in range(max_attempts):
        state = net.save_state()
        temp_mutate(net, temperature)
        new_score, new_acc = score_combined(net, perm, V)

        # Zone tracking
        if temperature < 0.5:
            zone_time['FOCUSED'] += 1
        elif temperature < 1.5:
            zone_time['NORMAL'] += 1
        elif temperature < 3.0:
            zone_time['WIDE'] += 1
        else:
            zone_time['EARTHQUAKE'] += 1

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)

            if temp_mode == 'temp_gradual':
                old_t = temperature
                temperature = max(0.3, temperature * 0.95)
                event_log.append((att, 'COOL', old_t, temperature, new_acc))
            elif temp_mode == 'temp_aggressive':
                old_t = temperature
                temperature = max(0.2, temperature * 0.90)
                event_log.append((att, 'COOL', old_t, temperature, new_acc))
        else:
            net.restore_state(state)
            stale += 1

            if temp_mode == 'temp_gradual':
                if stale % 200 == 0:
                    old_t = temperature
                    temperature = min(5.0, temperature * 1.3)
                    event_log.append((att, 'HEAT', old_t, temperature, acc))
            elif temp_mode == 'temp_aggressive':
                if stale % 200 == 0:
                    old_t = temperature
                    temperature = min(8.0, temperature * 1.5)
                    event_log.append((att, 'HEAT', old_t, temperature, acc))

        temp_history.append(temperature)
        acc_history.append(best_acc)

        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    return {
        'temp_history': temp_history,
        'acc_history': acc_history,
        'event_log': event_log,
        'zone_time': zone_time,
        'best_acc': best_acc,
        'kept': kept,
        'final_temp': temperature,
        'steps': len(temp_history),
    }


if __name__ == "__main__":
    for mode in ['temp_gradual', 'temp_aggressive']:
        print(f"\n{'=' * 70}")
        print(f"  {mode.upper()} — 64-class, seed=42, 8K attempts")
        print(f"{'=' * 70}")

        t0 = time.time()
        r = run_diagnostic(mode)
        elapsed = time.time() - t0

        th = np.array(r['temp_history'])
        ah = np.array(r['acc_history'])

        print(f"\n  Steps: {r['steps']}, Best acc: {r['best_acc']*100:.1f}%, "
              f"Kept: {r['kept']}, Time: {elapsed:.0f}s")

        # Temperature statistics
        print(f"\n  --- TEMPERATURE STATISTICS ---")
        print(f"  Min: {th.min():.4f}")
        print(f"  Max: {th.max():.4f}")
        print(f"  Mean: {th.mean():.4f}")
        print(f"  Median: {np.median(th):.4f}")
        print(f"  Std: {th.std():.4f}")
        print(f"  Final: {r['final_temp']:.4f}")

        # Percentiles
        print(f"\n  --- PERCENTILES ---")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"  P{p:2d}: {np.percentile(th, p):.4f}")

        # Zone time
        total = sum(r['zone_time'].values())
        print(f"\n  --- TIME IN ZONE ---")
        for zone, count in r['zone_time'].items():
            pct = count / total * 100 if total > 0 else 0
            bar = '#' * int(pct / 2)
            print(f"  {zone:<12s} {count:5d} ({pct:5.1f}%) {bar}")

        # Temperature histogram (ASCII)
        print(f"\n  --- TEMPERATURE HISTOGRAM ---")
        bins = [0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
        for i in range(len(bins) - 1):
            count = int(((th >= bins[i]) & (th < bins[i+1])).sum())
            pct = count / len(th) * 100
            bar = '#' * int(pct / 2)
            print(f"  [{bins[i]:4.1f}-{bins[i+1]:4.1f}) {count:5d} ({pct:5.1f}%) {bar}")

        # Floor/ceiling hits
        if mode == 'temp_gradual':
            floor, ceil = 0.3, 5.0
        else:
            floor, ceil = 0.2, 8.0
        floor_hits = int((th <= floor * 1.01).sum())
        ceil_hits = int((th >= ceil * 0.99).sum())
        print(f"\n  --- LIMIT HITS ---")
        print(f"  Floor ({floor}): {floor_hits} times ({floor_hits/len(th)*100:.1f}%)")
        print(f"  Ceiling ({ceil}): {ceil_hits} times ({ceil_hits/len(th)*100:.1f}%)")

        # Events summary
        cools = [e for e in r['event_log'] if e[1] == 'COOL']
        heats = [e for e in r['event_log'] if e[1] == 'HEAT']
        print(f"\n  --- EVENTS ---")
        print(f"  Cool events: {len(cools)}")
        print(f"  Heat events: {len(heats)}")

        if cools:
            cool_temps = [e[3] for e in cools]
            print(f"  Cool range: {min(cool_temps):.4f} - {max(cool_temps):.4f}")
        if heats:
            heat_temps = [e[3] for e in heats]
            print(f"  Heat range: {min(heat_temps):.4f} - {max(heat_temps):.4f}")

        # Convergence: last 1000 steps
        if len(th) > 1000:
            last_1k = th[-1000:]
            print(f"\n  --- LAST 1000 STEPS ---")
            print(f"  Mean: {last_1k.mean():.4f}, Std: {last_1k.std():.4f}")
            print(f"  Min: {last_1k.min():.4f}, Max: {last_1k.max():.4f}")
            zone_last = {'FOCUSED': 0, 'NORMAL': 0, 'WIDE': 0, 'EARTHQUAKE': 0}
            for t in last_1k:
                if t < 0.5:
                    zone_last['FOCUSED'] += 1
                elif t < 1.5:
                    zone_last['NORMAL'] += 1
                elif t < 3.0:
                    zone_last['WIDE'] += 1
                else:
                    zone_last['EARTHQUAKE'] += 1
            for zone, count in zone_last.items():
                if count > 0:
                    print(f"    {zone}: {count/10:.1f}%")

        # First 500 steps
        first_500 = th[:500]
        print(f"\n  --- FIRST 500 STEPS ---")
        print(f"  Mean: {first_500.mean():.4f}, Std: {first_500.std():.4f}")
        print(f"  Min: {first_500.min():.4f}, Max: {first_500.max():.4f}")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")
