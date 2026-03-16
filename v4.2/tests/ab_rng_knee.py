"""
RNG Knee Test: progressively worse random sources.
Where does SWG training break down?

Tier 1: PCG64         — gold standard PRNG
Tier 2: MT19937       — Python default (good)
Tier 3: xorshift32    — fast, decent, 32-bit
Tier 4: LCG           — old-school linear congruential (glibc)
Tier 5: LCG-short     — 16-bit LCG, terrible period (65536)
Tier 6: Counter mod P — deterministic counter with prime mod (barely random)
Tier 7: Alternating   — just alternates 0/1 (pathological)
"""
import numpy as np
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from graph import SelfWiringGraph


# ── RNG tiers (best to worst) ────────────────────────────────

class PCG64Rng:
    name = "PCG64 (gold)"
    tier = 1
    def __init__(self, seed=0xBEEF):
        self._rng = np.random.Generator(np.random.PCG64(seed))
    def randint(self, lo, hi):
        return int(self._rng.integers(lo, hi + 1))
    def random(self):
        return float(self._rng.random())

class MT19937Rng:
    name = "MT19937 (Python default)"
    tier = 2
    def __init__(self, seed=42):
        import random as _r
        self._r = _r.Random(seed)
    def randint(self, lo, hi):
        return self._r.randint(lo, hi)
    def random(self):
        return self._r.random()

class Xorshift32Rng:
    name = "xorshift32"
    tier = 3
    def __init__(self, seed=12345):
        self._state = seed & 0xFFFFFFFF or 1
    def _next(self):
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self._state = x & 0xFFFFFFFF
        return x
    def randint(self, lo, hi):
        return lo + (self._next() % (hi - lo + 1))
    def random(self):
        return self._next() / 0xFFFFFFFF

class LCGRng:
    """glibc LCG: state = (a*state + c) mod m. Period = 2^31."""
    name = "LCG-32 (glibc)"
    tier = 4
    def __init__(self, seed=12345):
        self._state = seed & 0x7FFFFFFF
    def _next(self):
        self._state = (1103515245 * self._state + 12345) & 0x7FFFFFFF
        return self._state
    def randint(self, lo, hi):
        return lo + (self._next() % (hi - lo + 1))
    def random(self):
        return self._next() / 0x7FFFFFFF

class LCG16Rng:
    """16-bit LCG. Period = 65536. Terrible."""
    name = "LCG-16 (period=65K)"
    tier = 5
    def __init__(self, seed=12345):
        self._state = seed & 0xFFFF
    def _next(self):
        self._state = (25173 * self._state + 13849) & 0xFFFF
        return self._state
    def randint(self, lo, hi):
        return lo + (self._next() % (hi - lo + 1))
    def random(self):
        return self._next() / 0xFFFF

class CounterModP:
    """Counter with prime modulus. Deterministic, no randomness —
    just cycles through residues. Tests if 'uniform coverage' alone is enough."""
    name = "Counter mod P (fake)"
    tier = 6
    def __init__(self, seed=0):
        self._state = seed
        self._p = 104729  # prime
    def _next(self):
        self._state += 1
        return self._state % self._p
    def randint(self, lo, hi):
        return lo + (self._next() % (hi - lo + 1))
    def random(self):
        return (self._next() % 10000) / 10000.0

class AlternatingRng:
    """Pathological: alternates between low and high values."""
    name = "Alternating 0/1 (broken)"
    tier = 7
    def __init__(self, seed=0):
        self._state = seed
    def _next(self):
        self._state += 1
        return self._state
    def randint(self, lo, hi):
        self._state += 1
        # Alternates between lo and hi
        return lo if self._state % 2 == 0 else hi
    def random(self):
        self._state += 1
        return 0.2 if self._state % 2 == 0 else 0.8


# ── Training with pluggable RNG ──────────────────────────────

def train_with_rng(seed, rng, V=16, max_attempts=3000, ticks=8):
    np.random.seed(seed)
    net = SelfWiringGraph(V)
    targets = np.arange(V)

    def evaluate():
        logits = net.forward_batch(ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1) == targets).mean()
        tp = probs[np.arange(V), targets].mean()
        return 0.5 * acc + 0.5 * tp

    def patched_mutate():
        if rng.random() < 0.35:
            net.intensity = np.int8(max(1, min(15, int(net.intensity) + (1 if rng.random() < 0.5 else -1))))
        if rng.random() < 0.2:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + rng.randint(-3, 3))))
        undo = []
        for _ in range(int(net.intensity)):
            if net.signal:
                _flip(undo)
            else:
                if net.grow:
                    _add(undo)
                else:
                    if rng.random() < 0.7:
                        _remove(undo)
                    else:
                        _rewire(undo)
        return undo

    def _add(undo):
        r, c = rng.randint(0, net.N-1), rng.randint(0, net.N-1)
        if r != c and net.mask[r, c] == 0:
            net.mask[r, c] = net.DRIVE if rng.randint(0, 1) else -net.DRIVE
            net.alive.append((r, c))
            net.alive_set.add((r, c))
            undo.append(('A', r, c))

    def _flip(undo):
        if net.alive:
            idx = rng.randint(0, len(net.alive)-1)
            r, c = net.alive[idx]
            net.mask[r, c] *= -1
            undo.append(('F', r, c))

    def _remove(undo):
        if net.alive:
            idx = rng.randint(0, len(net.alive)-1)
            r, c = net.alive[idx]
            old = net.mask[r, c]
            net.mask[r, c] = 0
            net.alive[idx] = net.alive[-1]
            net.alive.pop()
            net.alive_set.discard((r, c))
            undo.append(('R', r, c, old))

    def _rewire(undo):
        if net.alive:
            idx = rng.randint(0, len(net.alive)-1)
            r, c = net.alive[idx]
            nc = rng.randint(0, net.N-1)
            if nc != r and nc != c and net.mask[r, nc] == 0:
                old = net.mask[r, c]
                net.mask[r, c] = 0
                net.mask[r, nc] = old
                net.alive[idx] = (r, nc)
                net.alive_set.discard((r, c))
                net.alive_set.add((r, nc))
                undo.append(('W', r, c, nc))

    score = evaluate()
    best = score
    stale = 0

    t0 = time.perf_counter()
    for att in range(max_attempts):
        old_loss = int(net.loss_pct)
        undo = patched_mutate()
        new_score = evaluate()
        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1
            if rng.random() < 0.35:
                net.signal = np.int8(1 - int(net.signal))
            if rng.random() < 0.35:
                net.grow = np.int8(1 - int(net.grow))
        if best >= 0.99 or stale >= 2000:
            break

    return {
        'score': best,
        'gens': att + 1,
        'edges': net.count_connections(),
        'time': time.perf_counter() - t0,
    }


# ── Run all tiers ─────────────────────────────────────────────

SEEDS = [42, 137, 256, 314, 999]
V = 16

ALL_RNGS = [
    lambda: PCG64Rng(0xBEEF),
    lambda: MT19937Rng(42),
    lambda: Xorshift32Rng(12345),
    lambda: LCGRng(12345),
    lambda: LCG16Rng(12345),
    lambda: CounterModP(0),
    lambda: AlternatingRng(0),
]

print("=" * 74)
print(f"RNG KNEE TEST  |  V={V}, max=3000 gen, {len(SEEDS)} seeds")
print("=" * 74)

tier_results = []

for rng_factory in ALL_RNGS:
    rng_sample = rng_factory()
    scores = []
    times = []
    for seed in SEEDS:
        rng = rng_factory()
        res = train_with_rng(seed, rng, V=V)
        scores.append(res['score'])
        times.append(res['time'])

    avg = np.mean(scores) * 100
    std = np.std(scores) * 100
    avg_t = np.mean(times)
    tier_results.append((rng_sample.tier, rng_sample.name, avg, std, avg_t, scores))
    print(f"  Tier {rng_sample.tier}: {rng_sample.name:30s}  "
          f"avg={avg:5.1f}% ±{std:4.1f}%  ({avg_t:.2f}s)")

# ── Summary table ─────────────────────────────────────────────
print("\n" + "=" * 74)
print("ÖSSZESÍTÉS — Hol a Knee?")
print("=" * 74)

baseline = tier_results[0][2]  # PCG64 average
print(f"\n{'Tier':>4s}  {'RNG':30s}  {'Átlag':>7s}  {'±Std':>6s}  {'Δ vs PCG':>8s}  {'Bar'}")
print(f"{'─'*4}  {'─'*30}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*20}")

for tier, name, avg, std, avg_t, scores in tier_results:
    delta = avg - baseline
    bar_len = max(0, int(avg / 2))
    bar = "█" * bar_len + "░" * (30 - bar_len)
    marker = ""
    if abs(delta) > 5:
        marker = " ← KNEE?" if delta < -5 else ""
    if abs(delta) > 15:
        marker = " ← BROKEN"
    print(f"  {tier:2d}   {name:30s}  {avg:5.1f}%  ±{std:4.1f}%  {delta:+7.1f}%  {bar}{marker}")

# Find the knee
print("\n── Knee detection ──")
prev_avg = baseline
for tier, name, avg, std, avg_t, scores in tier_results[1:]:
    drop = prev_avg - avg
    if drop > 3:
        print(f"  ⚡ Tier {tier} ({name}): -{drop:.1f}% drop from previous tier")
    prev_avg = avg

worst_ok = None
for tier, name, avg, std, avg_t, scores in tier_results:
    if abs(avg - baseline) < 3:
        worst_ok = (tier, name)

if worst_ok:
    print(f"\n  ✦ CHEAPEST USABLE: Tier {worst_ok[0]} — {worst_ok[1]}")
    print(f"    Bármi Tier {worst_ok[0]}-ig ugyanúgy működik. Ennél olcsóbb nem kell.")
else:
    print(f"\n  ✦ Már Tier 2-nél is nagy a drop — PCG64 kell.")
