"""
A/B test: PCG64 (best PRNG) vs os.urandom (OS kernel entropy)
Same initial network, same targets — only the mutation RNG differs.
"""
import numpy as np
import os
import struct
import time
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from graph import SelfWiringGraph

# ── RNG wrappers ──────────────────────────────────────────────

class PCG64Rng:
    """Best known PRNG. Passes all BigCrush tests."""
    def __init__(self, seed):
        self._rng = np.random.Generator(np.random.PCG64(seed))
    def randint(self, lo, hi):
        return int(self._rng.integers(lo, hi + 1))
    def random(self):
        return float(self._rng.random())
    @property
    def name(self):
        return "PCG64 (PRNG)"

class OsUrandomRng:
    """OS kernel CSPRNG — ChaCha20 on Linux 6.x, fed by hardware entropy.
    This is as close to 'true random' as software can get."""
    def randint(self, lo, hi):
        span = hi - lo + 1
        # rejection sampling to avoid modulo bias
        bits_needed = (span - 1).bit_length()
        byte_count = max(1, (bits_needed + 7) // 8)
        mask = (1 << bits_needed) - 1
        while True:
            raw = int.from_bytes(os.urandom(byte_count), 'little') & mask
            if raw < span:
                return lo + raw
    def random(self):
        return struct.unpack('d', struct.pack('Q',
            (int.from_bytes(os.urandom(8), 'little') >> 12) | 0x3FF0000000000000))[0] - 1.0
    @property
    def name(self):
        return "os.urandom (kernel CSPRNG)"

# ── Patch network to use custom RNG ──────────────────────────

def train_with_rng(seed, rng, V=16, max_attempts=3000, ticks=8):
    """Train SWG using a specific RNG source for mutations."""
    # Fixed seed for identical starting network
    np.random.seed(seed)
    net = SelfWiringGraph(V)

    # Save initial state to verify identical start
    init_edges = net.count_connections()
    init_checksum = float(net.mask.sum())

    targets = np.arange(V)

    def evaluate():
        logits = net.forward_batch(ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1) == targets).mean()
        tp = probs[np.arange(V), targets].mean()
        return 0.5 * acc + 0.5 * tp

    # Override random calls in mutation to use our RNG
    def patched_mutate():
        # Intensity drift
        if rng.random() < 0.35:
            net.intensity = np.int8(max(1, min(15, int(net.intensity) + (1 if rng.random() < 0.5 else -1))))
        # Loss step
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

    # ── Training loop ──
    score = evaluate()
    best = score
    stale = 0
    history = [score]

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

        if (att + 1) % 100 == 0:
            history.append(best)

        if best >= 0.99 or stale >= 2000:
            break

    elapsed = time.perf_counter() - t0
    return {
        'rng': rng.name,
        'init_edges': init_edges,
        'init_checksum': init_checksum,
        'final_score': best,
        'generations': att + 1,
        'final_edges': net.count_connections(),
        'elapsed': elapsed,
        'history': history,
    }


# ── Run A/B ───────────────────────────────────────────────────

SEEDS = [42, 137, 256, 314, 999]
V = 16

print("=" * 70)
print(f"A/B TESZT: PCG64 vs os.urandom  |  V={V}, max=3000 gen, {len(SEEDS)} seed")
print("=" * 70)

results_a = []
results_b = []

for seed in SEEDS:
    print(f"\n── Seed {seed} ──")

    rng_a = PCG64Rng(seed=0xBEEF)  # fixed PRNG seed for mutation
    res_a = train_with_rng(seed, rng_a, V=V)
    print(f"  [A] {res_a['rng']:30s} → {res_a['final_score']*100:5.1f}%  "
          f"({res_a['generations']:4d} gen, {res_a['final_edges']:3d} edges, "
          f"{res_a['elapsed']:.2f}s)")

    rng_b = OsUrandomRng()  # kernel entropy — no seed possible
    res_b = train_with_rng(seed, rng_b, V=V)
    print(f"  [B] {res_b['rng']:30s} → {res_b['final_score']*100:5.1f}%  "
          f"({res_b['generations']:4d} gen, {res_b['final_edges']:3d} edges, "
          f"{res_b['elapsed']:.2f}s)")

    # Verify identical start
    assert res_a['init_edges'] == res_b['init_edges'], "HIBA: eltérő kiinduló hálózat!"
    assert abs(res_a['init_checksum'] - res_b['init_checksum']) < 1e-6, "HIBA: eltérő checksum!"
    print(f"  ✓ Azonos kiindulás: {res_a['init_edges']} él, checksum={res_a['init_checksum']:.2f}")

    results_a.append(res_a)
    results_b.append(res_b)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ÖSSZESÍTÉS")
print("=" * 70)

scores_a = [r['final_score'] for r in results_a]
scores_b = [r['final_score'] for r in results_b]
gens_a = [r['generations'] for r in results_a]
gens_b = [r['generations'] for r in results_b]
time_a = [r['elapsed'] for r in results_a]
time_b = [r['elapsed'] for r in results_b]

print(f"\n{'':30s} {'PCG64':>12s}  {'os.urandom':>12s}  {'Δ':>8s}")
print(f"{'─'*30} {'─'*12}  {'─'*12}  {'─'*8}")
print(f"{'Átlag pontosság':30s} {np.mean(scores_a)*100:11.2f}%  {np.mean(scores_b)*100:11.2f}%  {(np.mean(scores_a)-np.mean(scores_b))*100:+7.2f}%")
print(f"{'Std pontosság':30s} {np.std(scores_a)*100:11.2f}%  {np.std(scores_b)*100:11.2f}%")
print(f"{'Átlag generáció':30s} {np.mean(gens_a):11.0f}   {np.mean(gens_b):11.0f}   {np.mean(gens_a)-np.mean(gens_b):+7.0f}")
print(f"{'Átlag idő':30s} {np.mean(time_a):10.2f}s  {np.mean(time_b):10.2f}s  {np.mean(time_a)-np.mean(time_b):+7.2f}s")

print(f"\n{'Seed-enkénti eredmények':30s}")
for i, seed in enumerate(SEEDS):
    sa, sb = scores_a[i]*100, scores_b[i]*100
    winner = "A" if sa > sb else ("B" if sb > sa else "=")
    print(f"  Seed {seed:4d}:  A={sa:5.1f}%  B={sb:5.1f}%  → {'DÖNTETLEN' if winner == '=' else winner + ' nyer'}")

a_wins = sum(1 for a, b in zip(scores_a, scores_b) if a > b)
b_wins = sum(1 for a, b in zip(scores_a, scores_b) if b > a)
ties = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) < 1e-6)

print(f"\nA nyer: {a_wins}  |  B nyer: {b_wins}  |  Döntetlen: {ties}")
diff = abs(np.mean(scores_a) - np.mean(scores_b)) * 100
print(f"\nKONKLÚZIÓ: {'NINCS ÉRDEMI KÜLÖNBSÉG' if diff < 2.0 else 'VAN KÜLÖNBSÉG'} (Δ = {diff:.2f}%)")
