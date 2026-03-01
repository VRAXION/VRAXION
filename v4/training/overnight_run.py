"""Overnight single-run launcher with YAML config override.

Usage:
    python overnight_run.py --run-name run33a_baseline --bb-enabled false
    python overnight_run.py --run-name run33b_bb_l2norm --bb-enabled true --bb-gate-bias 0.0 --bb-scale 0.1 --bb-tau 4.0
    python overnight_run.py --run-name run33c_bb_conservative --bb-enabled true --bb-gate-bias -1.0

Modifies vraxion_config.yaml in-place before launching train.py,
then restores original config after training completes.
Results go to training_output/<run-name>/.
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


def main():
    parser = argparse.ArgumentParser(description='Overnight single-run launcher')
    parser.add_argument('--run-name', required=True, help='Run name (used as output subdir)')
    parser.add_argument('--steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--bb-enabled', type=str, default=None, help='true/false')
    parser.add_argument('--bb-gate-bias', type=float, default=None)
    parser.add_argument('--bb-scale', type=float, default=None)
    parser.add_argument('--bb-tau', type=float, default=None)
    args = parser.parse_args()

    # Save original config
    backup_path = CONFIG_PATH.with_suffix('.yaml.bak')
    shutil.copy2(CONFIG_PATH, backup_path)

    try:
        cfg = load_config()

        # Apply overrides
        if args.bb_enabled is not None:
            cfg['model']['bb_enabled'] = args.bb_enabled.lower() == 'true'
        if args.bb_gate_bias is not None:
            cfg['model']['bb_gate_bias'] = args.bb_gate_bias
        if args.bb_scale is not None:
            cfg['model']['bb_scale'] = args.bb_scale
        if args.bb_tau is not None:
            cfg['model']['bb_tau'] = args.bb_tau

        # Set output dir
        out_dir = V4_ROOT / 'training_output' / args.run_name
        cfg['training']['out_dir'] = str(out_dir)
        cfg['training']['steps'] = args.steps

        save_config(cfg)

        print(f'[OVERNIGHT] Starting {args.run_name}')
        print(f'  bb_enabled={cfg["model"].get("bb_enabled", False)}')
        if cfg['model'].get('bb_enabled', False):
            print(f'  bb_gate_bias={cfg["model"].get("bb_gate_bias", 0.0)}')
            print(f'  bb_scale={cfg["model"].get("bb_scale", 0.1)}')
            print(f'  bb_tau={cfg["model"].get("bb_tau", 4.0)}')
        print(f'  steps={args.steps}')
        print(f'  out_dir={out_dir}')
        print()

        # Run training
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=str(V4_ROOT),
        )

        if result.returncode != 0:
            print(f'\n[OVERNIGHT] {args.run_name} FAILED (exit code {result.returncode})')
        else:
            print(f'\n[OVERNIGHT] {args.run_name} COMPLETE')

        return result.returncode

    finally:
        # Restore original config
        shutil.copy2(backup_path, CONFIG_PATH)
        backup_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
