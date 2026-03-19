"""Profile streaming learning: where is the time spent?"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph

V = 27
ALPHA = 2.0
TICKS = 8

np.random.seed(42)
net = SelfWiringGraph(V)
N = net.N

# Fake input stream
chars_input = np.random.randint(0, V, 1000)

# Timers
t_forward = 0
t_argmax = 0
t_active = 0
t_update = 0
t_explore = 0

net.state *= 0
net.charge *= 0

for i in range(999):
    world = np.zeros(V, dtype=np.float32)
    world[chars_input[i]] = 1.0

    # FORWARD PASS
    t0 = time.perf_counter()
    act = net.state.copy()
    for t in range(TICKS):
        if t == 0:
            act[:V] = world
        raw = act @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        net.charge += raw
        total = np.abs(act).sum() + 1e-6
        act = np.maximum(net.charge - net.THRESHOLD, 0.0)
        act /= (1.0 + ALPHA * total)
        net.charge = np.clip(net.charge, -1.0, 1.0)
    net.state = act.copy()
    output = net.charge[net.out_start:net.out_start + V]
    t_forward += time.perf_counter() - t0

    # ARGMAX + COMPARE
    t0 = time.perf_counter()
    pred = np.argmax(output)
    actual = chars_input[i + 1]
    is_correct = (pred == actual)
    error_signal = 1.0 if is_correct else -1.0
    t_argmax += time.perf_counter() - t0

    # FIND ACTIVE NEURONS
    t0 = time.perf_counter()
    active_mask = act > 0.01
    active_idx = np.where(active_mask)[0]
    t_active += time.perf_counter() - t0

    # THREE-FACTOR UPDATE (vectorized)
    t0 = time.perf_counter()
    lr = 0.05
    if len(active_idx) > 0 and len(active_idx) < N // 2:
        # Vectorized outer product update
        act_active = act[active_idx]
        outer = np.outer(act_active, act_active) * (lr * error_signal)
        # Only update where mask is non-zero (existing connections)
        sub_mask = net.mask[np.ix_(active_idx, active_idx)]
        update_where = sub_mask != 0
        sub_mask[update_where] += outer[update_where]
        np.clip(sub_mask, -2.0, 2.0, out=sub_mask)
        net.mask[np.ix_(active_idx, active_idx)] = sub_mask
        # Zero diagonal
        np.fill_diagonal(net.mask, 0)
    t_update += time.perf_counter() - t0

    # EXPLORATION (add connection on error)
    t0 = time.perf_counter()
    if not is_correct and np.random.random() < 0.1 and len(active_idx) > 1:
        src = np.random.choice(active_idx)
        dst = np.random.choice(active_idx)
        if src != dst and net.mask[src, dst] == 0:
            net.mask[src, dst] = net.DRIVE * error_signal
            net.alive.append((src, dst))
            net.alive_set.add((src, dst))
    t_explore += time.perf_counter() - t0

total = t_forward + t_argmax + t_active + t_update + t_explore
print(f"Profile: 1000 chars, V={V} N={N}")
print(f"{'component':<20s} {'time':>8s} {'%':>6s}")
print("-" * 36)
for name, t in [('forward pass', t_forward), ('argmax+compare', t_argmax),
                ('find active', t_active), ('three-factor update', t_update),
                ('exploration', t_explore)]:
    print(f"{name:<20s} {t*1000:7.1f}ms {100*t/total:5.1f}%")
print(f"{'TOTAL':<20s} {total*1000:7.1f}ms")
print(f"\nPer char: {total/999*1000:.3f}ms")
print(f"Chars/sec: {999/total:.0f}")
print(f"Active neurons (avg): {np.mean(active_mask.sum()):.1f}/{N}")

# Now profile with LARGER N
for test_V in [64, 128]:
    np.random.seed(42)
    net2 = SelfWiringGraph(test_V)
    N2 = net2.N
    net2.state *= 0; net2.charge *= 0

    t0 = time.perf_counter()
    for i in range(200):
        world = np.zeros(test_V, dtype=np.float32)
        world[i % test_V] = 1.0
        act = net2.state.copy()
        for t in range(TICKS):
            if t == 0: act[:test_V] = world
            raw = act @ net2.mask
            np.nan_to_num(raw, copy=False)
            net2.charge += raw
            total_act = np.abs(act).sum() + 1e-6
            act = np.maximum(net2.charge - 0.5, 0.0)
            act /= (1.0 + ALPHA * total_act)
            net2.charge = np.clip(net2.charge, -1.0, 1.0)
        net2.state = act.copy()

        # Vectorized update
        active = np.where(act > 0.01)[0]
        if len(active) > 0 and len(active) < N2 // 2:
            act_a = act[active]
            outer = np.outer(act_a, act_a) * 0.05
            sub = net2.mask[np.ix_(active, active)]
            wh = sub != 0
            sub[wh] += outer[wh]
            np.clip(sub, -2.0, 2.0, out=sub)
            net2.mask[np.ix_(active, active)] = sub

    elapsed = time.perf_counter() - t0
    n_active = len(np.where(act > 0.01)[0])
    print(f"\nV={test_V} N={N2}: 200 chars = {elapsed*1000:.0f}ms "
          f"({elapsed/200*1000:.2f}ms/char, {200/elapsed:.0f} chars/s) "
          f"active={n_active}")
