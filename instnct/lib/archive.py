"""
Experiment archive — compact long-term storage for checkpoints.
Every experiment saves here for later inspection/visualization.

Usage in recipes:
    from lib.archive import save_experiment
    save_experiment(
        name="sweep_output_dim_od160",
        mask=mask, theta=theta, decay=decay, polarity=pol,
        config={'H': 256, 'in_dim': 64, 'out_dim': 160, ...},
        result={'best': 0.208, 'edges': 3250, 'accepts': 1090},
    )

Archive format: one .npz per experiment, sparse COO mask, compressed.
Index: archive/INDEX.json — metadata for all experiments.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime

ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "archive"
INDEX_PATH = ARCHIVE_DIR / "INDEX.json"


def save_experiment(name, mask, theta, decay, polarity=None,
                    freq=None, phase=None, rho=None,
                    config=None, result=None, bp_in=None, bp_out=None):
    """Save experiment checkpoint to archive with metadata."""
    ARCHIVE_DIR.mkdir(exist_ok=True)

    # Sparse COO for mask
    rows, cols = np.where(mask != 0)
    H = mask.shape[0]

    # Build npz payload
    data = {
        'H': np.int32(H),
        'rows': rows.astype(np.uint16 if H <= 65535 else np.uint32),
        'cols': cols.astype(np.uint16 if H <= 65535 else np.uint32),
        'theta': theta.astype(np.float16),  # half precision for compression
        'decay': decay.astype(np.float16),
    }
    if polarity is not None:
        data['polarity'] = np.asarray(polarity, dtype=np.int8)
    if freq is not None:
        data['freq'] = freq.astype(np.float16)
    if phase is not None:
        data['phase'] = phase.astype(np.float16)
    if rho is not None:
        data['rho'] = rho.astype(np.float16)
    if bp_in is not None:
        data['bp_in'] = bp_in.astype(np.float16)
    if bp_out is not None:
        data['bp_out'] = bp_out.astype(np.float16)

    # Save compressed
    fpath = ARCHIVE_DIR / f"{name}.npz"
    np.savez_compressed(fpath, **data)

    # Update index
    index = load_index()
    entry = {
        'name': name,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'H': int(H),
        'edges': int(len(rows)),
        'file': f"{name}.npz",
        'size_kb': round(fpath.stat().st_size / 1024, 1),
    }
    if config:
        entry['config'] = config
    if result:
        entry['result'] = result
    # Replace if name exists
    index = [e for e in index if e['name'] != name]
    index.append(entry)
    index.sort(key=lambda e: e['date'], reverse=True)

    with open(INDEX_PATH, 'w') as f:
        json.dump(index, f, indent=2, default=str)

    return fpath


def load_index():
    """Load archive index."""
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            return json.load(f)
    return []


def list_experiments():
    """Print all archived experiments."""
    index = load_index()
    if not index:
        print("Archive empty.")
        return
    print(f"{'Name':40s} {'Date':16s} {'H':>4} {'Edges':>6} {'Size':>6} {'Best':>6}")
    for e in index:
        best = e.get('result', {}).get('best', '?')
        if isinstance(best, float):
            best = f"{best*100:.1f}%"
        print(f"{e['name']:40s} {e['date']:16s} {e['H']:4d} {e['edges']:6d} "
              f"{e['size_kb']:5.1f}K {best:>6}")
