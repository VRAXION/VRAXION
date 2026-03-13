"""A/B test: replace vs additive write mode.

Runs two short training sessions with identical seeds and compares loss curves.
Usage:
    cd v4/ && python tools/ab_write_mode.py --steps 500
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

V4_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = V4_ROOT / 'config' / 'vraxion_config.yaml'
TRAIN_SCRIPT = V4_ROOT / 'training' / 'train.py'


def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_training(run_name, write_mode, steps, seed):
    """Modify config, run training, return exit code."""
    cfg = load_config()
    cfg['model']['write_mode'] = write_mode
    cfg['model']['N'] = 1  # single expert for fast iteration
    cfg['model']['hidden_dim'] = 256  # smaller for CPU speed
    cfg['model']['slot_dim'] = 64
    cfg['model']['M'] = 128
    cfg['training']['device'] = 'cpu'
    cfg['training']['cpu_threads'] = 4
    cfg['training']['batch_size'] = 16
    cfg['training']['seq_len'] = 64
    cfg['training']['steps'] = steps
    cfg['training']['out_dir'] = str(V4_ROOT / 'training_output' / run_name)
    cfg['training']['log_every'] = 5
    cfg['training']['save_every'] = steps  # only save at end
    cfg['training']['heartbeat_every'] = 50
    save_config(cfg)

    print(f'\n{"="*60}')
    print(f'  {run_name}: write_mode={write_mode}, steps={steps}, seed={seed}')
    print(f'{"="*60}\n')

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), '--seed', str(seed)],
        cwd=str(V4_ROOT),
    )
    return result.returncode


def compare_logs():
    """Load CSV logs and print comparison."""
    import csv

    print(f'\n{"="*60}')
    print('  COMPARISON: replace vs additive')
    print(f'{"="*60}\n')

    for mode in ['replace', 'additive']:
        log_dir = V4_ROOT / 'training_output' / f'ab_{mode}'
        csv_files = sorted(log_dir.glob('*.csv'))
        if not csv_files:
            print(f'  {mode}: no CSV log found')
            continue
        csv_path = csv_files[-1]
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            print(f'  {mode}: empty CSV')
            continue

        # Print key milestones
        milestones = [0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1]
        loss_key = 'loss' if 'loss' in rows[0] else list(rows[0].keys())[1]
        print(f'  {mode} (from {csv_path.name}):')
        for mi in milestones:
            if mi < len(rows):
                r = rows[mi]
                step = r.get('step', mi)
                loss = float(r.get(loss_key, 0))
                print(f'    step {step:>5}: loss = {loss:.6f}')

        # Final loss
        final = float(rows[-1].get(loss_key, 0))
        print(f'    FINAL loss: {final:.6f}')
        print()


def main():
    parser = argparse.ArgumentParser(description='A/B test: replace vs additive write mode')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Backup original config
    backup = CONFIG_PATH.with_suffix('.yaml.bak')
    shutil.copy2(CONFIG_PATH, backup)

    try:
        for mode in ['replace', 'additive']:
            rc = run_training(f'ab_{mode}', mode, args.steps, args.seed)
            if rc != 0:
                print(f'WARNING: {mode} run failed (exit {rc})')

        compare_logs()
    finally:
        # Restore original config
        shutil.copy2(backup, CONFIG_PATH)
        backup.unlink(missing_ok=True)
        print('Original config restored.')


if __name__ == '__main__':
    main()
