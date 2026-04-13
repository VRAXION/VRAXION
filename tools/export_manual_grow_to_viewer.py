#!/usr/bin/env python3
"""Convert target/c19_manual_grow/<head>/state.json files into brain_replay
viewer trace format (docs/pages/brain_replay/traces/<task>.json) and refresh
tasks.json.

For each baked head, replays the neurons in bake order and computes partial
ensemble train_acc/val_acc at each step so the viewer's learning curve and
neuron-add log are populated correctly.

Usage:
    python tools/export_manual_grow_to_viewer.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from manual_grow_explorer import gen_data
from c19_manual_explorer import ensemble_accuracy_c19

VIEWER_DIR = ROOT / "docs" / "pages" / "brain_replay"
VIEWER_TRACES = VIEWER_DIR / "traces"
VIEWER_TASKS = VIEWER_DIR / "tasks.json"
SOURCE_ROOT = ROOT / "target" / "c19_manual_grow"
DATA_SEED = 42


def round4(x):
    return round(float(x), 4)


def convert_head(head_name: str, state: dict):
    neurons = state.get("neurons", [])
    if not neurons:
        return None

    train, val, test, n_in = gen_data(head_name, DATA_SEED)
    n_per = len(train[0]) // 2

    events = []
    for i in range(1, len(neurons) + 1):
        partial = neurons[:i]
        tr_acc = ensemble_accuracy_c19(partial, train[0], train[1], n_in)
        va_acc = ensemble_accuracy_c19(partial, val[0], val[1], n_in)
        n = neurons[i - 1]
        events.append({
            "event": "neuron_added",
            "tick": i,
            "id": i - 1,
            "parents": list(n["parents"]),
            "weights": list(n["weights"]),
            "threshold": int(n["threshold"]),
            "alpha": round4(n["alpha"]),
            "train_acc": round4(tr_acc),
            "val_acc": round4(va_acc),
            "c_float": round4(n["c"]),
            "rho_float": round4(n["rho"]),
            "c_quant": round4(n["c"]),
            "rho_quant": round4(n["rho"]),
            "lut_min_dot": int(n["lut_min_dot"]),
            "lut_size": len(n["lut"]),
            "lut": [round4(x) for x in n["lut"]],
        })

    final_va = ensemble_accuracy_c19(neurons, val[0], val[1], n_in)
    final_te = ensemble_accuracy_c19(neurons, test[0], test[1], n_in)

    return {
        "schema": "c19_grower.v1",
        "c19_grower": True,
        "task": head_name,
        "data_seed": DATA_SEED,
        "search_seed": 0,
        "n_in": n_in,
        "n_per": n_per,
        "noise": 0.1,
        "events": events,
        "final": {
            "best_val_acc": round4(final_va),
            "best_test_acc": round4(final_te),
            "total_neurons": len(neurons),
            "max_depth": len(neurons),
            "stall_count": 0,
        },
    }


def main():
    VIEWER_TRACES.mkdir(parents=True, exist_ok=True)
    converted = []
    skipped = 0
    for state_path in sorted(SOURCE_ROOT.glob("*/state.json")):
        head_name = state_path.parent.name
        try:
            state = json.loads(state_path.read_text())
        except Exception as e:
            print(f"  SKIP {head_name}: read error {e}")
            skipped += 1
            continue
        trace = convert_head(head_name, state)
        if trace is None:
            skipped += 1
            continue
        out_path = VIEWER_TRACES / f"{head_name}.json"
        out_path.write_text(json.dumps(trace, indent=2))
        converted.append({
            "task": head_name,
            "best_val_acc": trace["final"]["best_val_acc"],
            "total_neurons": trace["final"]["total_neurons"],
            "max_depth": trace["final"]["max_depth"],
            "winner_seed": 0,
        })
        print(f"  {head_name}: {trace['final']['best_val_acc']:>5.1f}% val, "
              f"{trace['final']['total_neurons']} neurons -> {out_path.name}")

    existing = json.loads(VIEWER_TASKS.read_text()) if VIEWER_TASKS.exists() else {"tasks": []}
    existing_tasks = {t["task"]: t for t in existing.get("tasks", [])}
    converted_names = {t["task"] for t in converted}
    merged = [t for name, t in existing_tasks.items() if name not in converted_names] + converted
    merged.sort(key=lambda t: t["task"])

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": merged,
    }
    VIEWER_TASKS.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(converted)} traces, skipped {skipped} empty heads.")
    print(f"tasks.json now has {len(merged)} total entries.")


if __name__ == "__main__":
    main()
