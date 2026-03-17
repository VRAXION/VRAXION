"""Pain system sweep: 5 conditions x 3 seeds, V=64, budget=16k.
Tests whether error feedback injected mid-forward onto compute neurons helps mutation search.
Pain = diff (target - output) injected at tick T onto compute neurons [V..2V-1].
Neuromod = threshold/retention modulated by error magnitude."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

V = 64
BUDGET = 16000
STALE = 6000
SEEDS = [0, 42, 123]

CONDITIONS = {
    'baseline':   dict(pain_intensity=0.0, neuro_strength=0.0, pain_tick=4),
    'pain_input': dict(pain_intensity=0.5, neuro_strength=0.0, pain_tick=4),
    'neuromod':   dict(pain_intensity=0.0, neuro_strength=1.0, pain_tick=4),
    'pain_both':  dict(pain_intensity=0.5, neuro_strength=1.0, pain_tick=4),
    'pain_early': dict(pain_intensity=0.5, neuro_strength=0.0, pain_tick=2),
}


def forward_batch_pain(net, ticks=8, pain_tick=4, pain_intensity=0.0,
                       neuro_strength=0.0):
    V, N = net.V, net.N
    out_start = net.out_start
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    threshold = net.THRESHOLD

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)

        # Pain injection at pain_tick
        if t == pain_tick and (pain_intensity > 0 or neuro_strength > 0):
            # Read intermediate output
            mid = charges[:, out_start:out_start + V]
            e = np.exp(mid - mid.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            diff = np.eye(V, dtype=np.float32) - probs  # target - output

            # Inject pain onto compute neurons
            if pain_intensity > 0:
                charges[:, V:2*V] += diff * pain_intensity

            # Neuromodulation: lower threshold + higher retention
            if neuro_strength > 0:
                error_mag = np.abs(diff).mean()
                threshold = net.THRESHOLD * (1.0 - neuro_strength * error_mag)
                threshold = max(0.05, threshold)
                retain = float(net.retention) * (1.0 + neuro_strength * error_mag * 0.5)
                retain = min(0.99, retain)

        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - threshold, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

    return charges[:, out_start:out_start + V]


def evaluate_pain(net, targets, pain_tick=4, pain_intensity=0.0,
                  neuro_strength=0.0):
    logits = forward_batch_pain(net, ticks=8, pain_tick=pain_tick,
                                pain_intensity=pain_intensity,
                                neuro_strength=neuro_strength)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train_pain(net, targets, budget, pain_tick=4, pain_intensity=0.0,
               neuro_strength=0.0, stale_limit=6000):
    score = evaluate_pain(net, targets, pain_tick, pain_intensity, neuro_strength)
    best = score
    stale = 0

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate_pain(net, targets, pain_tick, pain_intensity, neuro_strength)

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if best >= 0.99 or stale >= stale_limit:
            break

    return best, att + 1


# --- Main ---
print(f"PAIN SYSTEM SWEEP | V={V} budget={BUDGET} stale={STALE}")
print(f"{'condition':<14s} {'seed':>4s}  {'score':>7s} {'steps':>6s} {'time':>6s}")
print("-" * 48)

results = {}

for name, kwargs in CONDITIONS.items():
    results[name] = []
    for seed in SEEDS:
        np.random.seed(seed)
        random.seed(seed)
        net = SelfWiringGraph(V)
        targets = np.arange(V)
        np.random.shuffle(targets)

        random.seed(seed * 1000 + 1)
        t0 = time.time()
        best, steps = train_pain(net, targets, BUDGET, stale_limit=STALE, **kwargs)
        elapsed = time.time() - t0
        results[name].append(best)
        print(f"{name:<14s} {seed:4d}  {best*100:6.1f}% {steps:6d} {elapsed:5.0f}s",
              flush=True)

# Summary
print(f"\n{'='*60}")
print(f"PAIN SWEEP SUMMARY | V={V} budget={BUDGET}")
print(f"{'condition':<14s} {'mean':>7s} {'std':>6s} {'vs_base':>8s}  per-seed")
print("-" * 60)
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
    print(f"{name:<14s} {m:6.1f}% {s:5.1f}pp {diff:+7.1f}pp  [{per}]")
print(f"\nBest: {best_name} ({best_mean:.1f}%)")
print(f"{'='*60}")
