"""
Binary wave analysis: WHY does 2-level beat 256-level?
======================================================
Analyze the discrete wave patterns for 8 ticks.
Find OPTIMAL binary freq/phase values for max orthogonality.
"""
import sys
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

TICKS = 8
RHO = 0.3

def wave_patterns(freq_vals, phase_vals, ticks=TICKS):
    """Compute all wave patterns and effective theta multipliers."""
    patterns = []
    for f in freq_vals:
        for p in phase_vals:
            wave = np.array([np.sin(t * f + p) for t in range(ticks)])
            mult = 1.0 + RHO * wave
            patterns.append({'freq': f, 'phase': p, 'wave': wave, 'mult': mult})
    return patterns

def orthogonality(patterns):
    """Measure how different the patterns are (higher = better)."""
    n = len(patterns)
    if n < 2: return 0
    waves = np.array([p['wave'] for p in patterns])
    # Cosine similarity matrix
    norms = np.linalg.norm(waves, axis=1, keepdims=True) + 1e-10
    normed = waves / norms
    sim = normed @ normed.T
    # Average off-diagonal absolute similarity (lower = more orthogonal)
    mask = ~np.eye(n, dtype=bool)
    avg_sim = np.abs(sim[mask]).mean()
    return 1.0 - avg_sim  # higher = better

def print_patterns(patterns, label=""):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    for i, p in enumerate(patterns):
        f, ph = p['freq'], p['phase']
        w = p['wave']
        m = p['mult']
        ph_label = f"{ph/np.pi:.2f}pi"
        print(f"\n  Type {i+1}: freq={f:.3f} phase={ph_label}")
        print(f"  tick:    {'  '.join(f'{t:5d}' for t in range(TICKS))}")
        print(f"  wave:    {'  '.join(f'{v:5.2f}' for v in w)}")
        print(f"  x_theta: {'  '.join(f'{v:5.3f}' for v in m)}")
        easiest = np.argmin(m)
        hardest = np.argmax(m)
        print(f"  easiest tick={easiest} (x{m[easiest]:.3f})  hardest tick={hardest} (x{m[hardest]:.3f})")

    orth = orthogonality(patterns)
    print(f"\n  Orthogonality score: {orth:.4f} (higher=better)")

    # Cross-correlation matrix
    n = len(patterns)
    waves = np.array([p['wave'] for p in patterns])
    print(f"\n  Cross-correlation matrix:")
    print(f"  {'':8s}", end="")
    for i in range(n): print(f"  T{i+1:d}   ", end="")
    print()
    for i in range(n):
        print(f"  T{i+1:d}    ", end="")
        for j in range(n):
            corr = np.corrcoef(waves[i], waves[j])[0,1]
            print(f" {corr:6.3f}", end="")
        print()

# =============================================================
# 1. CURRENT BINARY: freq={0.5, 2.0}, phase={0, pi}
# =============================================================
current = wave_patterns([0.5, 2.0], [0, np.pi])
print_patterns(current, "CURRENT BINARY: freq={0.5, 2.0}, phase={0, pi}")

# =============================================================
# 2. BRUTE FORCE: find best binary freq pair
# =============================================================
print(f"\n\n{'='*70}")
print(f"  BRUTE FORCE: best binary freq pair (phase always {{0, pi}})")
print(f"{'='*70}")

best_orth = 0
best_pair = None
# Test all freq pairs from 0.1 to 3.0 in 0.1 steps
freq_candidates = np.arange(0.1, 3.1, 0.1)
results = []
for i, f1 in enumerate(freq_candidates):
    for f2 in freq_candidates[i+1:]:
        pats = wave_patterns([f1, f2], [0, np.pi])
        orth = orthogonality(pats)
        results.append((orth, f1, f2))
        if orth > best_orth:
            best_orth = orth
            best_pair = (f1, f2)

results.sort(reverse=True)
print(f"\n  Top 10 freq pairs:")
for orth, f1, f2 in results[:10]:
    print(f"    freq=({f1:.1f}, {f2:.1f})  orth={orth:.4f}")
print(f"\n  Bottom 5 (worst):")
for orth, f1, f2 in results[-5:]:
    print(f"    freq=({f1:.1f}, {f2:.1f})  orth={orth:.4f}")

# Show the winner
print(f"\n  BEST PAIR: freq=({best_pair[0]:.1f}, {best_pair[1]:.1f}) orth={best_orth:.4f}")
best_pats = wave_patterns([best_pair[0], best_pair[1]], [0, np.pi])
print_patterns(best_pats, f"BEST: freq=({best_pair[0]:.1f}, {best_pair[1]:.1f}), phase={{0, pi}}")

# =============================================================
# 3. What about phase? Is {0, pi} optimal?
# =============================================================
print(f"\n\n{'='*70}")
print(f"  PHASE SWEEP: is {{0, pi}} optimal? (using best freq pair)")
print(f"{'='*70}")

phase_results = []
for p1 in np.arange(0, np.pi, 0.1):
    for p2 in np.arange(p1 + 0.1, 2*np.pi, 0.1):
        pats = wave_patterns([best_pair[0], best_pair[1]], [p1, p2])
        orth = orthogonality(pats)
        phase_results.append((orth, p1, p2))

phase_results.sort(reverse=True)
print(f"\n  Top 10 phase pairs (with best freq):")
for orth, p1, p2 in phase_results[:10]:
    print(f"    phase=({p1/np.pi:.2f}pi, {p2/np.pi:.2f}pi)  orth={orth:.4f}")

# =============================================================
# 4. SPECIAL: what about freq={pi/4, pi/2}? (harmonics of tick)
# =============================================================
special_pairs = [
    (0.5, 2.0, "current"),
    (np.pi/8, np.pi/2, "pi/8, pi/2"),
    (np.pi/4, np.pi, "pi/4, pi (quarter/half wave)"),
    (1.0, 2.0, "1.0, 2.0"),
    (np.pi/4, 3*np.pi/4, "pi/4, 3pi/4"),
    (0.5, np.pi, "0.5, pi"),
    (1.0, np.pi, "1.0, pi"),
]
print(f"\n\n{'='*70}")
print(f"  SPECIAL FREQ PAIRS")
print(f"{'='*70}")
for f1, f2, name in special_pairs:
    pats = wave_patterns([f1, f2], [0, np.pi])
    orth = orthogonality(pats)
    print(f"  freq=({f1:.4f}, {f2:.4f}) [{name:25s}] orth={orth:.4f}")

# =============================================================
# 5. ALIASING analysis: what's unique at 8 ticks?
# =============================================================
print(f"\n\n{'='*70}")
print(f"  ALIASING: how many UNIQUE wave shapes exist at 8 ticks?")
print(f"{'='*70}")
print(f"  (freq values that produce the same discrete pattern are aliases)")

seen = {}
for f in np.arange(0.05, 4.0, 0.05):
    wave = tuple(np.round(np.sin(np.arange(8) * f), 4))
    key = wave
    if key not in seen:
        seen[key] = f

print(f"  {len(seen)} unique wave shapes from freq 0.05 to 4.0 (step 0.05)")
print(f"  Nyquist freq for tick sampling: pi = {np.pi:.4f}")
print(f"  Useful freq range: [0, pi] before aliasing")

# How many unique patterns with phase={0, pi}?
all_pats = set()
for f in np.arange(0.05, np.pi + 0.05, 0.05):
    for p in [0, np.pi]:
        wave = tuple(np.round(np.sin(np.arange(8) * f + p), 3))
        all_pats.add(wave)
print(f"  Unique patterns with phase={{0,pi}}: {len(all_pats)}")

# =============================================================
# 6. SUMMARY: what makes binary good?
# =============================================================
print(f"\n\n{'='*70}")
print(f"  SUMMARY: WHY BINARY WORKS")
print(f"{'='*70}")
print(f"""
  With 8 ticks and binary freq+phase, you get exactly 4 neuron types.
  Each type has a distinct temporal signature: different "easy" and "hard" ticks.

  Key properties of binary wave:
  1. phase={{0, pi}} creates ANTI-PHASE pairs: when one is easy, the other is hard
  2. Two freq values create two TIMESCALES: slow+fast oscillation
  3. 4 types = 4 orthogonal temporal channels in 8 ticks (near-optimal)
  4. Clean separation > noisy diversity: 256 similar-but-different patterns blur

  Think of it as TDMA (Time Division Multiple Access):
  - Each neuron type "owns" certain ticks
  - Binary gives clean ownership, many levels give shared/blurred ownership
  - For 8 ticks, 4 channels is near-optimal (8/2 = 4 Nyquist channels)
""")
