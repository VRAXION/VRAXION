"""
Checkpoint helper — automatic save/load for all experiment scripts.
Import and use: save_checkpoint(), load_checkpoint()

Saves to: instnct/recipes/checkpoints/{name}_{step}.npz
Always includes: qdata, theta, channel/tick_weights, polarity, metadata.
"""
import os, json, time
from pathlib import Path
import numpy as np

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def save_checkpoint(name, step, *, qdata, theta, polarity,
                    channel=None, tick_weights=None,
                    best_eval=0.0, config=None, extra=None):
    """Save experiment state. Called automatically at eval points + end."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        'qdata': qdata,
        'theta': theta,
        'polarity': polarity,
        'step': np.array(step),
        'best_eval': np.array(best_eval),
        'timestamp': np.array(time.time()),
    }
    if channel is not None:
        data['channel'] = channel
    if tick_weights is not None:
        data['tick_weights'] = tick_weights
    if extra is not None:
        for k, v in extra.items():
            data[k] = np.asarray(v)

    # Save latest (overwritten each time)
    latest_path = CHECKPOINT_DIR / f"{name}_latest.npz"
    np.savez_compressed(latest_path, **data)

    # Save milestone at key steps
    if step % 500 == 0 or step <= 50:
        milestone_path = CHECKPOINT_DIR / f"{name}_step{step}.npz"
        np.savez_compressed(milestone_path, **data)

    # Save metadata as JSON (human readable)
    meta = {
        'name': name, 'step': step,
        'best_eval': round(float(best_eval), 4),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if config is not None:
        meta['config'] = config
    meta_path = CHECKPOINT_DIR / f"{name}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return str(latest_path)


def load_checkpoint(name, step=None):
    """Load checkpoint. step=None loads latest."""
    if step is not None:
        path = CHECKPOINT_DIR / f"{name}_step{step}.npz"
    else:
        path = CHECKPOINT_DIR / f"{name}_latest.npz"

    if not path.exists():
        return None

    data = dict(np.load(path, allow_pickle=False))
    return data


def list_checkpoints(name=None):
    """List available checkpoints."""
    if not CHECKPOINT_DIR.exists():
        return []
    pattern = f"{name}_*.npz" if name else "*.npz"
    return sorted(CHECKPOINT_DIR.glob(pattern))
