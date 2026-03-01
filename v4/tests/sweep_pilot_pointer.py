"""
Pilot Pulse Pointer — A/B Sweep
================================
4 runs × 1000 steps on WikiText-103:
  1. Baseline: pointer_mode=sequential (ptr += 1)
  2. Pilot A:  pointer_mode=pilot, max_jump=128 (conservative)
  3. Pilot B:  pointer_mode=pilot, max_jump=256 (moderate)
  4. Pilot C:  pointer_mode=pilot, max_jump=512 (aggressive)

Usage:
  cd v4
  python -u -X utf8 tests/sweep_pilot_pointer.py
"""

import sys
import os
import subprocess
import time
import shutil
from pathlib import Path

# ── paths ──
V4_ROOT = Path(__file__).parent.parent
CONFIG_PATH = V4_ROOT / 'config' / 'vraxion_config.yaml'
TRAIN_SCRIPT = V4_ROOT / 'training' / 'train.py'
SWEEP_DIR = V4_ROOT / 'training_output' / 'sweep_pilot_pointer'

STEPS = 1000

CONFIGS = [
    {'name': 'baseline_sequential', 'pointer_mode': 'sequential', 'pilot_max_jump': 256},
    {'name': 'pilot_jump128',       'pointer_mode': 'pilot',      'pilot_max_jump': 128},
    {'name': 'pilot_jump256',       'pointer_mode': 'pilot',      'pilot_max_jump': 256},
    {'name': 'pilot_jump512',       'pointer_mode': 'pilot',      'pilot_max_jump': 512},
]

def patch_yaml(pointer_mode: str, pilot_max_jump: int):
    """Patch vraxion_config.yaml with pointer settings."""
    text = CONFIG_PATH.read_text(encoding='utf-8')

    # Replace pointer_mode line
    import re
    text = re.sub(
        r'(pointer_mode:\s*)\S+',
        f'\\g<1>{pointer_mode}',
        text
    )
    # Replace pilot_max_jump line
    text = re.sub(
        r'(pilot_max_jump:\s*)\S+',
        f'\\g<1>{pilot_max_jump}',
        text
    )
    CONFIG_PATH.write_text(text, encoding='utf-8')

def run_training(name: str, pointer_mode: str, pilot_max_jump: int):
    """Run one training config."""
    out_dir = SWEEP_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  RUN: {name}')
    print(f'  pointer_mode={pointer_mode}, pilot_max_jump={pilot_max_jump}')
    print(f'  steps={STEPS}, out={out_dir}')
    print(f'{"="*60}\n')

    # Patch config
    patch_yaml(pointer_mode, pilot_max_jump)

    # Save a copy of the config used
    shutil.copy2(CONFIG_PATH, out_dir / 'vraxion_config.yaml')

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, '-u', '-X', 'utf8', str(TRAIN_SCRIPT),
         '--steps', str(STEPS),
         '--out', str(out_dir),
         '--device', 'cuda'],
        cwd=str(V4_ROOT),
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    print(f'\n  {name} finished in {elapsed:.0f}s (exit={result.returncode})')
    return result.returncode

def summarize():
    """Read train_log.csv from each run and print final metrics."""
    print(f'\n{"="*60}')
    print(f'  SWEEP SUMMARY')
    print(f'{"="*60}\n')
    print(f'{"Name":<25} {"Best Loss":>10} {"Best Acc":>10} {"Time":>8}')
    print('-' * 55)

    for cfg in CONFIGS:
        csv_path = SWEEP_DIR / cfg['name'] / 'train_log.csv'
        if not csv_path.exists():
            print(f'{cfg["name"]:<25} {"MISSING":>10}')
            continue

        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print(f'{cfg["name"]:<25} {"EMPTY":>10}')
            continue

        # Find best loss and corresponding acc
        best_loss = float('inf')
        best_acc = 0.0
        for row in rows:
            loss = float(row.get('eval_loss', row.get('loss', '999')))
            acc = float(row.get('eval_masked_acc', row.get('masked_acc', '0')))
            if loss < best_loss:
                best_loss = loss
                best_acc = acc

        # Last row for final acc
        last = rows[-1]
        final_acc = float(last.get('eval_masked_acc', last.get('masked_acc', '0')))

        print(f'{cfg["name"]:<25} {best_loss:>10.4f} {final_acc*100:>9.1f}%')

    print()


if __name__ == '__main__':
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # Save original config
    original_yaml = CONFIG_PATH.read_text(encoding='utf-8')

    try:
        for cfg in CONFIGS:
            rc = run_training(cfg['name'], cfg['pointer_mode'], cfg['pilot_max_jump'])
            if rc != 0:
                print(f'  WARNING: {cfg["name"]} failed with exit code {rc}')
    finally:
        # Restore original config
        CONFIG_PATH.write_text(original_yaml, encoding='utf-8')
        print('\nConfig restored to original.')

    summarize()
