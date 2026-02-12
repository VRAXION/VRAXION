"""Quick test of frequency analysis code without launching Streamlit."""

import sys
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Golden Draft")

from tools.live_dashboard import parse_log
import numpy as np
from scipy import signal

# Parse the probe log
log_path = "S:/AI/work/VRAXION_DEV/Golden Draft/logs/probe/probe_live.log"
print(f"Loading log: {log_path}")
df = parse_log(log_path)

print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check if we have enough data
freq_df = df[["step", "loss"]].dropna()
print(f"\nRows with loss data: {len(freq_df)}")

if len(freq_df) >= 200:
    print("[OK] Sufficient data for frequency analysis")

    # Test the frequency analysis code
    if len(freq_df) > 300:
        stride = max(1, len(freq_df) // 300)
        freq_df = freq_df.iloc[::stride].reset_index(drop=True)

    step_arr = freq_df["step"].values
    loss_arr = freq_df["loss"].values
    loss_detrended = signal.detrend(loss_arr)

    print(f"\nAfter subsampling: {len(step_arr)} points")
    print(f"Loss range: [{loss_arr.min():.4f}, {loss_arr.max():.4f}]")
    print(f"Loss mean: {loss_arr.mean():.4f} ± {loss_arr.std():.4f}")

    # Test periodogram
    window_size = min(100, len(loss_detrended) // 3)
    print(f"\nPeriodogram window size: {window_size}")

    if window_size >= 50:
        freqs, power = signal.periodogram(loss_detrended[:window_size], fs=1.0)
        mask = (freqs >= 0.01) & (freqs <= 0.5)
        print(f"Frequency bins: {mask.sum()}")

        # Find dominant frequency
        dominant_idx = np.argmax(power[mask])
        dominant_freq = freqs[mask][dominant_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0
        print(f"Dominant frequency: {dominant_freq:.4f} cycles/step (period ~{dominant_period:.1f} steps)")

    # Test peak detection
    peaks, _ = signal.find_peaks(loss_detrended)
    print(f"\nPeaks detected: {len(peaks)}")

    if len(peaks) >= 2:
        spacings = np.diff(peaks)
        print(f"Peak spacing: {spacings.mean():.2f} ± {spacings.std():.2f} steps")

    # Test autocorrelation
    max_lag = min(50, len(loss_detrended) // 4)
    if max_lag >= 10:
        acorr = np.correlate(loss_detrended, loss_detrended, mode='full')
        acorr = acorr[len(acorr)//2:]
        acorr = acorr / acorr[0]
        lags = np.arange(1, max_lag + 1)
        acorr_vals = acorr[1:max_lag+1]

        # Find strongest positive lag
        positive_mask = acorr_vals > 0
        if positive_mask.any():
            strongest_lag = lags[positive_mask][np.argmax(acorr_vals[positive_mask])]
            strongest_corr = acorr_vals[positive_mask].max()
            print(f"Strongest autocorrelation: lag {strongest_lag} (r={strongest_corr:+.3f})")

    # Test Butterworth filter
    if len(loss_arr) >= 50:
        nyquist = 0.5
        cutoff = 0.05
        order = 2
        b, a = signal.butter(order, cutoff / nyquist, btype='low')
        trend = signal.filtfilt(b, a, loss_arr)
        oscillation = loss_arr - trend

        print(f"\nOscillation amplitude: {oscillation.std():.4f}")
        print(f"Trend range: [{trend.min():.4f}, {trend.max():.4f}]")

    print("\n[OK] All frequency analysis functions work correctly!")

else:
    print("[FAIL] Not enough data (need >=200 points)")
