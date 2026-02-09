"""AGC Live Probe — true no-AGC test at full update scale.

Modes (set PROBE_MODE env var):
  fullscale  (default) : AGC OFF + VRX_SCALE_MIN=1.0  — the real test
  agc_on               : AGC ON, defaults             — Part C control
  locked_low           : AGC OFF, VRX_SCALE_MIN=0.01  — Part C agc-off (confounded)

Run:  python tools/_scratch/agc_live_probe.py
Stop: Ctrl-C any time; heartbeat prints every 10 steps to stderr.
"""
from __future__ import annotations

import os
import sys

# --- Mode selection (before any vraxion imports) ---
MODE = os.environ.get("PROBE_MODE", "fullscale").strip().lower()

# Common env for all modes
_COMMON = {
    "VRX_SYNTH": "1",
    "VRX_SYNTH_ONCE": "1",
    "VRX_SYNTH_MODE": "assoc_clean",
    "VRX_MAX_STEPS": "2000",
    "VRX_RING_LEN": "64",
    "VRX_SLOT_DIM": "32",
    "VAR_COMPUTE_DEVICE": "cpu",
    "VRX_HEARTBEAT_STEPS": "10",
    "VRX_MODULAR_SAVE": "0",
    "VRX_NAN_GUARD": "0",
    "VRX_DISABLE_SYNC": "1",
}

_MODES = {
    "fullscale":  {"VRX_AGC_ENABLED": "0", "VRX_SCALE_MIN": "1.0"},
    "agc_on":     {"VRX_AGC_ENABLED": "1"},
    "locked_low": {"VRX_AGC_ENABLED": "0", "VRX_SCALE_MIN": "0.01"},
}

if MODE not in _MODES:
    print(f"Unknown PROBE_MODE={MODE!r}. Choose: {', '.join(_MODES)}", file=sys.stderr)
    sys.exit(1)

for k, v in {**_COMMON, **_MODES[MODE]}.items():
    os.environ[k] = v

# --- Path bootstrap (match gpu_learning_campaign pattern) ---
_DRAFTR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _DRAFTR)
for c in [os.path.join(os.path.dirname(_DRAFTR), "Golden Code"), r"S:\AI\Golden Code"]:
    if os.path.isdir(c) and c not in sys.path:
        sys.path.insert(0, c)
        break

import torch  # noqa: E402
from vraxion.instnct.absolute_hallway import AbsoluteHallway  # noqa: E402
from tools import instnct_train_steps, instnct_data  # noqa: E402

# Do NOT silence log — heartbeat prints naturally to stderr

print(f"[agc_live_probe] mode={MODE}  env overrides: {_MODES[MODE]}", file=sys.stderr)

loader, num_classes, collate = instnct_data.get_seq_mnist_loader()
model = AbsoluteHallway(input_dim=1, num_classes=num_classes, ring_len=64, slot_dim=32)
print(f"[agc_live_probe] model ready  classes={num_classes}  starting 2000 steps...", file=sys.stderr)

result = instnct_train_steps.train_steps(model, loader, 2000, "assoc_clean", "absolute_hallway")

print(f"[agc_live_probe] DONE  loss_slope={result.get('loss_slope')}  "
      f"final_scale={getattr(model, 'update_scale', '?')}", file=sys.stderr)
