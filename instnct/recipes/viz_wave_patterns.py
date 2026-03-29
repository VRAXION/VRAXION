"""
Visualize wave patterns for all tested phase pairs.
ASCII art: 8 ticks, effective theta multiplier per type.
"""
import numpy as np
import sys
sys.stdout.reconfigure(line_buffering=True)

TICKS = 8
RHO = 0.3
FREQ = 1.0  # fixed in the phase test

configs = [
    ('A', '{0, pi}', [0, np.pi], 9.5),
    ('B', '{0.80pi, 1.27pi}', [0.80*np.pi, 1.27*np.pi], 11.1),
    ('C', '{0.32pi, 0.83pi}', [0.32*np.pi, 0.83*np.pi], 15.6),
    ('D', '{pi/4, 3pi/4}', [np.pi/4, 3*np.pi/4], 14.5),
    ('E', '{pi/2, 3pi/2}', [np.pi/2, 3*np.pi/2], 13.5),
]

def ascii_wave(values, width=50, lo=-1.0, hi=1.0, char='#', empty='.'):
    """Single row ASCII plot."""
    out = []
    for v in values:
        pos = int((v - lo) / (hi - lo) * (width - 1))
        pos = max(0, min(width - 1, pos))
        row = [empty] * width
        # zero line
        zero_pos = int((0 - lo) / (hi - lo) * (width - 1))
        row[zero_pos] = '|'
        row[pos] = char
        out.append(''.join(row))
    return out

for mk, label, phases, best in configs:
    diff = (phases[1] - phases[0]) / np.pi
    print(f"\n{'='*72}")
    print(f"  {mk}: {label}  diff={diff:.2f}pi  BEST={best}%")
    print(f"{'='*72}")

    types = []
    for pi, p in enumerate(phases):
        wave = np.array([np.sin(t * FREQ + p) for t in range(TICKS)])
        mult = 1.0 + RHO * wave
        types.append({'phase': p, 'wave': wave, 'mult': mult, 'idx': pi})

    # Correlation between the two types
    corr = np.corrcoef(types[0]['wave'], types[1]['wave'])[0,1]
    print(f"  Correlation: r={corr:.4f}")

    # ASCII plot: wave values for both types across 8 ticks
    print(f"\n  Wave value [-1, +1]:  '0'=type0  'X'=type1  '|'=zero")
    print(f"  {'tick':>6s}  {'t0_wave':>7s} {'t1_wave':>7s}  plot")
    for t in range(TICKS):
        w0 = types[0]['wave'][t]
        w1 = types[1]['wave'][t]
        # Draw both on same line
        width = 50
        lo, hi = -1.0, 1.0
        row = list('.' * width)
        zero_pos = int((0 - lo) / (hi - lo) * (width - 1))
        row[zero_pos] = '|'
        p0 = int((w0 - lo) / (hi - lo) * (width - 1))
        p1 = int((w1 - lo) / (hi - lo) * (width - 1))
        p0 = max(0, min(width-1, p0))
        p1 = max(0, min(width-1, p1))
        if p0 == p1:
            row[p0] = '*'  # overlap
        else:
            row[p0] = '0'
            row[p1] = 'X'
        print(f"  tick {t}  {w0:+7.3f} {w1:+7.3f}  {''.join(row)}")

    # Effective theta multiplier timeline
    print(f"\n  Theta multiplier [0.7, 1.3]:  '0'=type0  'X'=type1")
    print(f"  {'tick':>6s}  {'t0_mult':>7s} {'t1_mult':>7s}  plot  {'easy/hard':>20s}")
    for t in range(TICKS):
        m0 = types[0]['mult'][t]
        m1 = types[1]['mult'][t]
        width = 50
        lo, hi = 0.65, 1.35
        row = list('.' * width)
        one_pos = int((1.0 - lo) / (hi - lo) * (width - 1))
        row[one_pos] = '|'
        p0 = int((m0 - lo) / (hi - lo) * (width - 1))
        p1 = int((m1 - lo) / (hi - lo) * (width - 1))
        p0 = max(0, min(width-1, p0))
        p1 = max(0, min(width-1, p1))
        if p0 == p1:
            row[p0] = '*'
        else:
            row[p0] = '0'
            row[p1] = 'X'
        # Who fires easier?
        if abs(m0 - m1) < 0.05:
            who = "~same"
        elif m0 < m1:
            who = "0 easy, X hard"
        else:
            who = "X easy, 0 hard"
        print(f"  tick {t}  x{m0:.3f}  x{m1:.3f}  {''.join(row)}  {who}")

    # Summary: how many ticks does each type "own" (lower threshold)?
    t0_easy = sum(1 for t in range(TICKS) if types[0]['mult'][t] < types[1]['mult'][t] - 0.02)
    t1_easy = sum(1 for t in range(TICKS) if types[1]['mult'][t] < types[0]['mult'][t] - 0.02)
    tied = TICKS - t0_easy - t1_easy
    print(f"\n  Tick ownership: type0={t0_easy} ticks, type1={t1_easy} ticks, tied={tied}")
    print(f"  Separation quality: {abs(t0_easy - t1_easy)} imbalance (0=perfect)")

# Overall ranking
print(f"\n\n{'='*72}")
print(f"  SUMMARY: phase diff vs accuracy")
print(f"{'='*72}")
print(f"  {'Mode':>4s}  {'Phase diff':>10s}  {'Corr':>7s}  {'Best':>6s}  {'Note'}")
for mk, label, phases, best in sorted(configs, key=lambda x: -x[3]):
    diff = (phases[1] - phases[0]) / np.pi
    corr = np.corrcoef(
        [np.sin(t * FREQ + phases[0]) for t in range(TICKS)],
        [np.sin(t * FREQ + phases[1]) for t in range(TICKS)]
    )[0,1]
    note = ""
    if abs(diff - 0.5) < 0.05: note = "<-- ~pi/2 (quadrature)"
    elif abs(diff - 1.0) < 0.05: note = "<-- pi (anti-phase)"
    print(f"  {mk:>4s}  {diff:>8.2f}pi  {corr:>+7.3f}  {best:>5.1f}%  {note}")
