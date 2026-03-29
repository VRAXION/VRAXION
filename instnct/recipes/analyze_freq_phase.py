"""
Freq + Phase distribution analysis from all saved checkpoints
==============================================================
Load every .npz, extract freq/phase arrays, print stats + ASCII histogram.
Goal: see where these converge after training.
"""
import sys, os
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[1]

def ascii_hist(values, bins=20, width=50, label=""):
    """Print ASCII histogram."""
    counts, edges = np.histogram(values, bins=bins)
    mx = max(counts) if max(counts) > 0 else 1
    print(f"\n  {label}  (N={len(values)}, mean={values.mean():.4f}, std={values.std():.4f})")
    print(f"  range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"  median={np.median(values):.4f}, Q1={np.percentile(values,25):.4f}, Q3={np.percentile(values,75):.4f}")
    for i in range(bins):
        bar = '#' * int(counts[i] / mx * width)
        lo, hi = edges[i], edges[i+1]
        print(f"  [{lo:7.3f},{hi:7.3f}) {counts[i]:4d} |{bar}")

def wave_profile(freq, phase, ticks=8):
    """For each neuron, compute the wave values across all ticks."""
    # wave[tick] = sin(tick * freq + phase) for each neuron
    tick_arr = np.arange(ticks).reshape(-1, 1)  # (ticks, 1)
    waves = np.sin(tick_arr * freq.reshape(1, -1) + phase.reshape(1, -1))  # (ticks, H)
    return waves

if __name__ == "__main__":
    # Find all checkpoints
    dirs = [
        ROOT_DIR / "checkpoints",
        ROOT_DIR / "archive",
    ]

    all_freq = []
    all_phase = []
    checkpoint_data = []

    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.npz")):
            try:
                with np.load(f, allow_pickle=True) as data:
                    files = list(data.files)
                    if 'freq' in files and 'phase' in files:
                        freq = np.array(data['freq'], dtype=np.float32)
                        phase = np.array(data['phase'], dtype=np.float32)
                        H = len(freq)
                        checkpoint_data.append({
                            'name': f.stem,
                            'path': str(f),
                            'freq': freq,
                            'phase': phase,
                            'H': H,
                        })
                        all_freq.append(freq)
                        all_phase.append(phase)
                        print(f"  OK: {f.stem:40s} H={H:4d} freq=[{freq.min():.3f},{freq.max():.3f}] mean={freq.mean():.3f} "
                              f"phase=[{phase.min():.3f},{phase.max():.3f}] mean={phase.mean():.3f}")
                    else:
                        print(f"  --: {f.stem:40s} (no freq/phase, keys={files[:5]}...)")
            except Exception as e:
                print(f"  ERR: {f.stem}: {e}")

    if not all_freq:
        print("\nNo checkpoints with freq/phase found!")
        sys.exit(1)

    # Aggregate stats
    all_freq_flat = np.concatenate(all_freq)
    all_phase_flat = np.concatenate(all_phase)

    print(f"\n{'='*70}")
    print(f"  AGGREGATED FROM {len(checkpoint_data)} CHECKPOINTS")
    print(f"{'='*70}")

    ascii_hist(all_freq_flat, bins=20, label="FREQ (all checkpoints)")
    ascii_hist(all_phase_flat, bins=20, label="PHASE (all checkpoints)")

    # Per-checkpoint details
    print(f"\n{'='*70}")
    print(f"  PER-CHECKPOINT DETAIL")
    print(f"{'='*70}")

    for cp in checkpoint_data:
        freq = cp['freq']
        phase = cp['phase']
        name = cp['name']
        H = cp['H']

        print(f"\n--- {name} (H={H}) ---")

        # Freq distribution
        ascii_hist(freq, bins=15, label=f"FREQ [{name}]")
        ascii_hist(phase, bins=15, label=f"PHASE [{name}]")

        # Wave analysis: effective modulation across 8 ticks
        waves = wave_profile(freq, phase, ticks=8)  # (8, H)

        # For each tick, what's the mean/std of wave values?
        print(f"\n  Wave profile across ticks (rho=0.3 -> theta modulation range):")
        print(f"  {'tick':>4s}  {'wave_mean':>9s}  {'wave_std':>8s}  {'eff_mod_mean':>12s}  {'eff_mod_range':>14s}")
        for t in range(8):
            w = waves[t]
            mod = 1.0 + 0.3 * w  # effective multiplier on theta
            print(f"  {t:4d}  {w.mean():9.4f}  {w.std():8.4f}  "
                  f"x{mod.mean():5.3f}  [{mod.min():.3f}, {mod.max():.3f}]")

        # How many neurons have "similar" freq? (cluster analysis)
        # Round to 1 decimal to see clusters
        freq_rounded = np.round(freq, 1)
        unique_freqs, freq_counts = np.unique(freq_rounded, return_counts=True)
        top_freqs = sorted(zip(freq_counts, unique_freqs), reverse=True)[:10]
        print(f"\n  Top freq clusters (rounded to 0.1):")
        for cnt, fv in top_freqs:
            pct = cnt / H * 100
            print(f"    freq={fv:.1f}: {cnt:3d} neurons ({pct:.1f}%)")

        # Phase distribution: uniform or clustered?
        phase_rounded = np.round(phase / (np.pi/4)) * (np.pi/4)  # round to pi/4
        unique_phases, phase_counts = np.unique(phase_rounded, return_counts=True)
        top_phases = sorted(zip(phase_counts, unique_phases), reverse=True)[:8]
        print(f"\n  Top phase clusters (rounded to pi/4):")
        for cnt, pv in top_phases:
            pct = cnt / H * 100
            label_p = f"{pv/np.pi:.2f}pi" if abs(pv) > 0.01 else "0"
            print(f"    phase={label_p:>8s} ({pv:.3f}): {cnt:3d} neurons ({pct:.1f}%)")

        # Correlation between freq and phase?
        if len(freq) > 10:
            corr = np.corrcoef(freq, phase)[0, 1]
            print(f"\n  freq-phase correlation: r={corr:.4f}")

    # Summary recommendation
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION RECOMMENDATION")
    print(f"{'='*70}")
    freq_all = all_freq_flat
    phase_all = all_phase_flat

    # Check if freq converges to a narrow range
    freq_iqr = np.percentile(freq_all, 75) - np.percentile(freq_all, 25)
    phase_iqr = np.percentile(phase_all, 75) - np.percentile(phase_all, 25)

    print(f"  FREQ:  mean={freq_all.mean():.4f} std={freq_all.std():.4f} IQR={freq_iqr:.4f} range=[{freq_all.min():.4f},{freq_all.max():.4f}]")
    print(f"  PHASE: mean={phase_all.mean():.4f} std={phase_all.std():.4f} IQR={phase_iqr:.4f} range=[{phase_all.min():.4f},{phase_all.max():.4f}]")

    if freq_all.std() < 0.15:
        print(f"  >> FREQ may be FIX-able: very narrow spread (std={freq_all.std():.4f})")
    elif freq_iqr < 0.3:
        print(f"  >> FREQ int4 candidate: moderate spread (IQR={freq_iqr:.4f})")
    else:
        print(f"  >> FREQ wide spread: needs float or fine-grained int")

    if phase_all.std() < 0.5:
        print(f"  >> PHASE may be FIX-able: narrow spread (std={phase_all.std():.4f})")
    elif phase_iqr < 1.0:
        print(f"  >> PHASE int4 candidate: moderate spread (IQR={phase_iqr:.4f})")
    else:
        print(f"  >> PHASE wide spread: needs diversity, int4 minimum")

    print(f"{'='*70}")
