"""
INSTNCT Unified Visualization Generator
========================================
Reads an .npz checkpoint (+ optional training JSON) and produces a single
self-contained HTML file with three tabs: MASKS, GRAPH, TRAINING.

Usage:
  python gen_unified_viz.py                          # auto-detect checkpoint
  python gen_unified_viz.py --checkpoint PATH        # specific checkpoint
  python gen_unified_viz.py --demo                   # generate with dummy data
  python gen_unified_viz.py --export-json             # also write payload.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "instnct_viz.html"
TEMPLATE_MARKER = "/*__DATA__*/ null"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate unified INSTNCT visualization.")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--training-data", type=Path, default=None)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--export-json", action="store_true")
    p.add_argument("--demo", action="store_true", help="Generate with random dummy data")
    return p.parse_args()


def _step_key(path: Path) -> tuple[int, str]:
    m = re.search(r"_step(\d+)\.npz$", path.name)
    return (int(m.group(1)), path.name) if m else (-1, path.name)


def pick_default_checkpoint() -> Path:
    ckpt_dir = ROOT / "checkpoints"
    for pattern in ["english_1024n_step*.npz", "english_1024n*.npz", "instnct_*.npz", "*.npz"]:
        candidates = sorted(ckpt_dir.glob(pattern), key=_step_key)
        if candidates:
            return candidates[-1]
    raise FileNotFoundError("No checkpoint found under instnct/checkpoints")


def load_checkpoint(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        H = int(d["H"]) if "H" in files else int(np.asarray(d["theta"]).shape[0])
        V = int(d["V"]) if "V" in files else 256
        _sdr_dim = int(d['sdr_dim']) if 'sdr_dim' in files else 0
        _output_dim = int(d['output_dim']) if 'output_dim' in files else 0
        _phi_overlap = bool(d['phi_overlap']) if 'phi_overlap' in files else False

        rows = np.asarray(d["rows"], dtype=np.int32) if "rows" in files else np.array([], dtype=np.int32)
        cols = np.asarray(d["cols"], dtype=np.int32) if "cols" in files else np.array([], dtype=np.int32)
        if "vals" in files:
            keep = np.asarray(d["vals"]) != 0
            rows, cols = rows[keep], cols[keep]
        # Remove self-edges
        no_self = rows != cols
        rows, cols = rows[no_self], cols[no_self]

        def field(name, default_val=0.0):
            if name in files:
                return np.asarray(d[name], dtype=np.float32)
            return np.full(H, default_val, dtype=np.float32)

        theta = field("theta", 15.0)
        decay = field("decay", 1.0)
        polarity = np.asarray(d["polarity"], dtype=np.int8) if "polarity" in files else np.ones(H, dtype=np.int8)
        freq = field("freq", 1.0)
        phase = field("phase", 0.0)
        rho = field("rho", 0.3)

        ip = np.asarray(d["input_projection"], dtype=np.float32) if "input_projection" in files else None
        op = np.asarray(d["output_projection"], dtype=np.float32) if "output_projection" in files else None

    # Degrees
    out_deg = np.zeros(H, dtype=np.int32)
    in_deg = np.zeros(H, dtype=np.int32)
    for r in rows:
        out_deg[r] += 1
    for c in cols:
        in_deg[c] += 1

    # I/O energy
    input_energy = np.abs(ip).sum(axis=0) if ip is not None else np.zeros(H)
    output_energy = np.abs(op).sum(axis=1) if op is not None else np.zeros(H)

    def top_n(arr, n):
        return list(map(int, np.argsort(arr)[::-1][:n]))

    n_io = min(max(V // 4, 16), H // 4)

    return {
        "meta": {
            "H": H, "V": V,
            "edges": int(len(rows)),
            "density": round(len(rows) / max(H * H, 1), 6),
            "checkpoint": path.name,
            "sdr_dim": _sdr_dim,
            "output_dim": _output_dim,
            "phi_overlap": _phi_overlap,
        },
        "mask": {
            "rows": rows.tolist(),
            "cols": cols.tolist(),
        },
        "scalars": {
            name: {
                "values": [round(float(x), 5) for x in arr],
                "min": round(float(arr.min()), 5),
                "max": round(float(arr.max()), 5),
                "mean": round(float(arr.mean()), 5),
                "std": round(float(arr.std()), 5),
            }
            for name, arr in [
                ("theta", theta), ("decay", decay),
                ("polarity", polarity.astype(np.float32)),
                ("freq", freq), ("phase", phase), ("rho", rho),
            ]
        },
        "degrees": {
            "out": out_deg.tolist(),
            "in": in_deg.tolist(),
        },
        "io_energy": {
            "input": [round(float(x), 4) for x in input_energy],
            "output": [round(float(x), 4) for x in output_energy],
            "top_input": top_n(input_energy, n_io),
            "top_output": top_n(output_energy, n_io),
        },
    }


def generate_demo_data(H=256, V=32, density=0.04, seed=42) -> dict:
    rng = np.random.RandomState(seed)
    n_edges = int(H * H * density)
    rows = rng.randint(0, H, size=n_edges).astype(np.int32)
    cols = rng.randint(0, H, size=n_edges).astype(np.int32)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    # Deduplicate
    edges = list(set(zip(rows.tolist(), cols.tolist())))
    rows = np.array([e[0] for e in edges], dtype=np.int32)
    cols = np.array([e[1] for e in edges], dtype=np.int32)

    theta = rng.uniform(1, 15, H).astype(np.float32)
    decay = rng.uniform(0.5, 2.0, H).astype(np.float32)
    polarity = np.ones(H, dtype=np.int8)
    polarity[rng.rand(H) < 0.1] = -1
    freq = rng.uniform(0.5, 2.0, H).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, H).astype(np.float32)
    rho = rng.uniform(0, 1, H).astype(np.float32)

    out_deg = np.zeros(H, dtype=np.int32)
    in_deg = np.zeros(H, dtype=np.int32)
    for r in rows:
        out_deg[r] += 1
    for c in cols:
        in_deg[c] += 1

    input_energy = rng.rand(H).astype(np.float32)
    output_energy = rng.rand(H).astype(np.float32)
    n_io = min(max(V // 4, 8), H // 4)

    def top_n(arr, n):
        return list(map(int, np.argsort(arr)[::-1][:n]))

    training = [
        {
            "step": s, "eval": round(5 + 15 * (1 - math.exp(-s / 300)), 2),
            "edges": int(50 + s * 0.8), "A": s // 3, "T": s // 8, "D": s // 5,
            "theta_m": round(8.0 - s * 0.002, 4), "decay_m": round(0.16 + s * 0.0001, 4),
            "sps": 2.5, "time": s * 2,
        }
        for s in range(50, 1050, 50)
    ]

    return {
        "meta": {"H": H, "V": V, "edges": len(rows), "density": round(len(rows) / (H * H), 6), "checkpoint": "demo_random", "sdr_dim": 0, "output_dim": 0, "phi_overlap": False},
        "mask": {"rows": rows.tolist(), "cols": cols.tolist()},
        "scalars": {
            name: {
                "values": [round(float(x), 5) for x in arr],
                "min": round(float(arr.min()), 5), "max": round(float(arr.max()), 5),
                "mean": round(float(arr.mean()), 5), "std": round(float(arr.std()), 5),
            }
            for name, arr in [
                ("theta", theta), ("decay", decay), ("polarity", polarity.astype(np.float32)),
                ("freq", freq), ("phase", phase), ("rho", rho),
            ]
        },
        "degrees": {"out": out_deg.tolist(), "in": in_deg.tolist()},
        "io_energy": {
            "input": [round(float(x), 4) for x in input_energy],
            "output": [round(float(x), 4) for x in output_energy],
            "top_input": top_n(input_energy, n_io),
            "top_output": top_n(output_energy, n_io),
        },
        "training": training,
    }


def load_training_data(path: Path) -> list | None:
    if path and path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_all_archive_presets() -> dict:
    """Load all archive checkpoints as lightweight presets for embedding in HTML."""
    archive_dir = ROOT / "archive"
    index_path = archive_dir / "INDEX.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        index = json.load(f)

    presets = {}
    for entry in index:
        fpath = archive_dir / entry['file']
        if not fpath.exists():
            continue
        try:
            with np.load(fpath, allow_pickle=True) as d:
                H = int(d['H'])
                rows = np.array(d['rows'], dtype=np.int32)
                cols = np.array(d['cols'], dtype=np.int32)
                theta = np.array(d['theta'], dtype=np.float32)
                decay = np.array(d['decay'], dtype=np.float32)
                polarity = np.array(d['polarity'], dtype=np.int8) if 'polarity' in d else np.ones(H, dtype=np.int8)

                # Scalars
                scalars = {}
                for name, arr in [('theta', theta), ('decay', decay), ('polarity', polarity.astype(np.float32))]:
                    scalars[name] = {
                        'values': [round(float(x), 4) for x in arr],
                        'min': round(float(arr.min()), 4),
                        'max': round(float(arr.max()), 4),
                    }
                for extra in ['freq', 'phase', 'rho']:
                    if extra in d:
                        arr = np.array(d[extra], dtype=np.float32)
                        scalars[extra] = {
                            'values': [round(float(x), 4) for x in arr],
                            'min': round(float(arr.min()), 4),
                            'max': round(float(arr.max()), 4),
                        }

                # Degrees
                out_deg = np.zeros(H, dtype=np.int32)
                in_deg = np.zeros(H, dtype=np.int32)
                for r, c in zip(rows, cols):
                    out_deg[r] += 1
                    in_deg[c] += 1

                preset = {
                    'meta': {
                        'H': H,
                        'V': 256,
                        'edges': int(len(rows)),
                        'density': round(len(rows) / max(H * H, 1), 6),
                        'checkpoint': entry['name'],
                        'sdr_dim': entry.get('config', {}).get('sdr_dim', 0),
                        'output_dim': entry.get('config', {}).get('output_dim', 0),
                        'phi_overlap': entry.get('config', {}).get('phi_overlap', False),
                    },
                    'mask': {'rows': rows.tolist(), 'cols': cols.tolist()},
                    'scalars': scalars,
                    'degrees': {'out': out_deg.tolist(), 'in': in_deg.tolist()},
                    'label': entry['name'],
                    'date': entry.get('date', ''),
                    'result': entry.get('result', {}),
                    'config': entry.get('config', {}),
                }
                presets[entry['name']] = preset
        except Exception as e:
            print(f"  Warning: skip {entry['name']}: {e}")
    return presets


def build_html(payload: dict) -> str:
    template = (Path(__file__).parent / "unified_viz_template.html").read_text(encoding="utf-8")
    data_json = json.dumps(payload, separators=(",", ":"))
    return template.replace(TEMPLATE_MARKER, data_json)


def main():
    args = parse_args()

    if args.demo:
        payload = generate_demo_data()
        print(f"Generated demo data: H={payload['meta']['H']}, edges={payload['meta']['edges']}")
    else:
        try:
            ckpt = args.checkpoint or pick_default_checkpoint()
        except FileNotFoundError:
            print("No checkpoint found. Use --demo for dummy data.")
            return
        print(f"Loading: {ckpt}")
        payload = load_checkpoint(ckpt)

    # Training data
    if "training" not in payload or payload.get("training") is None:
        td_path = args.training_data
        if td_path is None:
            td_path = ROOT / "training_live_data.json"
        payload["training"] = load_training_data(td_path)

    # Archive presets — all checkpoints embedded for instant switching
    presets = load_all_archive_presets()
    if presets:
        payload["presets"] = presets
        print(f"Embedded {len(presets)} archive presets")

    if args.export_json:
        json_path = args.output.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"Exported JSON: {json_path}")

    html = build_html(payload)
    args.output.write_text(html, encoding="utf-8")
    print(f"Written: {args.output} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
