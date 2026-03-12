"""Overnight training watchdog — runs alongside train.py.

Checks CSV every 60s. Logs warnings. Does NOT restart or kill anything.
Run: python training/watchdog.py

Output: training_output/watchdog.log
"""
import time
import csv
import sys
from pathlib import Path
from datetime import datetime

V4_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = V4_ROOT / 'training_output' / 'train_log.csv'
LOG_PATH = V4_ROOT / 'training_output' / 'watchdog.log'
CHECK_INTERVAL = 60  # seconds

# Thresholds
ALPHA_WARN = 0.21        # near soft floor
RING_NORM_WARN = 8000    # getting large
GRAD_NORM_WARN = 9.0     # near clip threshold
STALL_MINUTES = 15       # no new CSV row for this long = stall


def log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def read_last_row():
    if not CSV_PATH.exists():
        return None
    with open(CSV_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main():
    log('Watchdog started')
    log(f'CSV: {CSV_PATH}')
    log(f'Interval: {CHECK_INTERVAL}s')
    log('---')

    last_step = 0
    last_step_time = time.time()
    best_loss = float('inf')
    nan_total = 0
    check_count = 0

    while True:
        try:
            row = read_last_row()
            if row is None:
                log('WARNING: no CSV data yet')
                time.sleep(CHECK_INTERVAL)
                continue

            step = int(row['step'])
            loss = float(row['masked_loss'])
            acc = float(row['masked_acc'])
            alpha = float(row['alpha_0_mean'])
            ring_norm = float(row['ring_norm'])
            grad_norm = float(row['grad_norm'])
            lr = float(row['lr'])

            check_count += 1
            now = time.time()

            # Track progress
            if step > last_step:
                last_step_time = now
                last_step = step

            if loss < best_loss:
                best_loss = loss

            # Stall detection
            stall_min = (now - last_step_time) / 60
            if stall_min > STALL_MINUTES:
                log(f'ALERT: Training stalled! No new step for {stall_min:.0f} min (last step={step})')

            # Warnings
            warnings = []
            if alpha <= ALPHA_WARN:
                warnings.append(f'alpha={alpha:.3f} (near floor)')
            if ring_norm > RING_NORM_WARN:
                warnings.append(f'ring_norm={ring_norm:.0f} (high)')
            if grad_norm > GRAD_NORM_WARN:
                warnings.append(f'grad_norm={grad_norm:.2f} (near clip)')
            if grad_norm == 0 and step > 200:
                nan_total += 1
                warnings.append(f'NaN detected (total={nan_total})')

            # Periodic status (every 5 min)
            if check_count % 5 == 1:
                log(f'step={step}  loss={loss:.4f}  acc={acc*100:.1f}%  '
                    f'alpha={alpha:.3f}  ring={ring_norm:.0f}  '
                    f'grad={grad_norm:.2f}  lr={lr:.6f}  best={best_loss:.4f}')

            # Print warnings
            for w in warnings:
                log(f'WARNING: {w}  (step={step})')

        except Exception as e:
            log(f'ERROR: {e}')

        time.sleep(CHECK_INTERVAL)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('Watchdog stopped (Ctrl+C)')
        sys.exit(0)
