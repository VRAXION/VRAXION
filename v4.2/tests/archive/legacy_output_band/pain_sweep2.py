"""Pain sweep v2: dedicated pain neurons + user ideas (bar, everyN).
BUGFIX: previous sweeps used eye(V) as target but targets are SHUFFLED.
All networks use NV_RATIO=4 (N=4V) for fair comparison.
Layout: input[0:V], pain[V:2V], compute[2V:3V], output[3V:4V].
Pain injected at tick 2 (inline, most stable from sweep 1)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

V = 64
BUDGET = 100000
STALE = 15000
SEEDS = [0, 42, 123]
LOG_EVERY = 10000
PAIN_TICK = 2

CONDITIONS = {
    'baseline':   dict(pain_mode='none',      pain_intensity=0.0),
    'dedicated':  dict(pain_mode='dedicated',  pain_intensity=0.3),
    'bar':        dict(pain_mode='bar',        pain_intensity=0.5),
    'everyN':     dict(pain_mode='everyN',     pain_intensity=0.1),
}


def make_net(V, seed):
    """Create 4V network with pain zone."""
    old_ratio = SelfWiringGraph.NV_RATIO
    SelfWiringGraph.NV_RATIO = 4
    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(V)
    SelfWiringGraph.NV_RATIO = old_ratio
    return net


def forward_pain(net, targets, ticks=8, pain_tick=PAIN_TICK,
                 pain_mode='none', pain_intensity=0.0):
    V, N = net.V, net.N
    out_start = net.out_start  # 3V for NV_RATIO=4
    pain_start = V
    pain_end = 2 * V

    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    threshold = net.THRESHOLD

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)

        if t == pain_tick and pain_mode != 'none':
            # Read intermediate output
            mid = charges[:, out_start:out_start + V]
            e = np.exp(mid - mid.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)

            if pain_mode == 'dedicated':
                # Full diff vector on pain zone — FIXED target matrix
                target_matrix = np.zeros((V, V), dtype=np.float32)
                target_matrix[np.arange(V), targets[:V]] = 1.0
                diff = target_matrix - probs
                charges[:, pain_start:pain_end] += diff * pain_intensity

            elif pain_mode == 'bar':
                # Equalizer bar: error magnitude → spatial (thermometer)
                # More error → more pain neurons active, concentrated on one side
                correct_probs = probs[np.arange(V), targets[:V]]
                errors = 1.0 - correct_probs  # 0=perfect, 1=total failure
                D = pain_end - pain_start
                for i in range(V):
                    num_active = max(1, int(errors[i] * D))
                    charges[i, pain_start:pain_start + num_active] += pain_intensity

            elif pain_mode == 'everyN':
                # Inject error magnitude into every 4th neuron across whole network
                correct_probs = probs[np.arange(V), targets[:V]]
                errors = (1.0 - correct_probs).reshape(V, 1)  # (V, 1)
                charges[:, ::4] += errors * pain_intensity

        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - threshold, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

    return charges[:, out_start:out_start + V]


def evaluate(net, targets, pain_mode='none', pain_intensity=0.0):
    logits = forward_pain(net, targets, pain_mode=pain_mode,
                          pain_intensity=pain_intensity)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train(net, targets, budget, pain_mode='none', pain_intensity=0.0,
          stale_limit=STALE, log_every=LOG_EVERY):
    score = evaluate(net, targets, pain_mode, pain_intensity)
    best = score
    stale = 0
    trajectory = [(0, best)]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate(net, targets, pain_mode, pain_intensity)

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if (att + 1) % log_every == 0:
            trajectory.append((att + 1, best))

        if best >= 0.99 or stale >= stale_limit:
            break

    if (att + 1) % log_every != 0:
        trajectory.append((att + 1, best))

    return best, att + 1, trajectory


# --- Main ---
print(f"PAIN SWEEP v2 (BUGFIX) | V={V} N={4*V} budget={BUDGET} pain_tick={PAIN_TICK}")
print(f"Layout: input[0:{V}] pain[{V}:{2*V}] compute[{2*V}:{3*V}] output[{3*V}:{4*V}]")
print(f"{'condition':<12s} {'seed':>4s}  {'score':>7s} {'steps':>7s} {'time':>6s}  trajectory")
print("-" * 95)

results = {}
trajectories = {}

for name, kwargs in CONDITIONS.items():
    results[name] = []
    trajectories[name] = []
    for seed in SEEDS:
        net = make_net(V, seed)
        targets = np.arange(V)
        np.random.shuffle(targets)

        random.seed(seed * 1000 + 1)
        t0 = time.time()
        best, steps, traj = train(net, targets, BUDGET, **kwargs)
        elapsed = time.time() - t0
        results[name].append(best)
        trajectories[name].append(traj)

        traj_str = " → ".join(f"{b*100:.1f}" for _, b in traj)
        print(f"{name:<12s} {seed:4d}  {best*100:6.1f}% {steps:7d} {elapsed:5.0f}s  {traj_str}",
              flush=True)

# Summary
print(f"\n{'='*70}")
print(f"PAIN SWEEP v2 SUMMARY | V={V} N={4*V}")
print(f"{'condition':<12s} {'mean':>7s} {'std':>6s} {'vs_base':>8s}  per-seed")
print("-" * 70)
base_mean = np.mean(results['baseline']) * 100
best_mean = -1
best_name = ''
for name in CONDITIONS:
    scores = results[name]
    m = np.mean(scores) * 100
    s = np.std(scores) * 100
    diff = m - base_mean
    per = " ".join(f"{sc*100:.1f}" for sc in scores)
    if m > best_mean:
        best_mean = m
        best_name = name
    print(f"{name:<12s} {m:6.1f}% {s:5.1f}pp {diff:+7.1f}pp  [{per}]")

# Trajectory table
print(f"\n{'='*70}")
print("TRAJECTORY (best % at each 10k checkpoint, averaged across seeds)")
print(f"{'step':>7s}", end="")
for name in CONDITIONS:
    print(f" {name:>12s}", end="")
print()
print("-" * (7 + 13 * len(CONDITIONS)))

checkpoints = [i * LOG_EVERY for i in range(1, BUDGET // LOG_EVERY + 1)]
for cp in checkpoints:
    print(f"{cp:7d}", end="")
    for name in CONDITIONS:
        vals = []
        for traj in trajectories[name]:
            best_at_cp = traj[0][1]
            for step, b in traj:
                if step <= cp:
                    best_at_cp = b
            vals.append(best_at_cp)
        avg = np.mean(vals) * 100
        print(f" {avg:11.1f}%", end="")
    print()

print(f"\nBest: {best_name} ({best_mean:.1f}%)")
print(f"{'='*70}")
