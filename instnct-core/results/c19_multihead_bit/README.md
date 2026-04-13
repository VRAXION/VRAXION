# C19 Multi-Head Bit Decomposition Results

This directory contains the baked c19 ensembles for the multi-head bit-decomposition exploration of the grid3 task family. Each subdirectory is one **bit head** — an independent c19 ensemble that classifies one output pixel of a grid3 task.

## Contents

| Subdirectory | Bit heads baked | Neurons total | Status |
|---|---|---|---|
| `grid3_center` | 1 | 2 | Reference single-task bake |
| `grid3_copy_bit_0..8` | 9/9 | 18 | Complete (2 neurons per head) |
| `grid3_invert_bit_0..8` | 9/9 | 9 | Complete (1 neuron per head — `f(0)=1` collapse) |
| `grid3_shift_right_bit_0..8` | 9/9 | 18 | Complete (2 neurons per head) |
| `grid3_reflect_h_bit_0..5` | 6/9 | 12 | Partial (loop interrupted at bit 6) |
| **TOTAL** | **34 heads** | **59 neurons** | — |

## Schema

Each `<head>/state.json` follows the c19 manual grow schema:

```json
{
  "task": "grid3_copy_bit_0",
  "activation": "c19",
  "neurons": [
    {
      "parents": [0, 1, 6],
      "weights": [1, -1, -1],
      "threshold": 0,
      "c": 3.0,
      "rho": 2.0,
      "lut": [0.0, -0.37, -0.37, 0.0, 0.96, 0.96, 0.0],
      "lut_min_dot": -3,
      "alpha": 0.4598966812126782
    }
  ]
}
```

## Provenance

- **Generator**: `tools/overnight_build_step.py` (intelligent top-1 picker + exhaustive 1-3 parent enumeration)
- **Activation**: c19 with learnable `c`, `rho`
- **Voting**: AdaBoost soft continuous (`score = Σ alpha_i * lut_i(dot_i)`)
- **Data seed**: 42, **noise**: 0.1, **n_per**: 200 train / 200 val / 200 test
- **Selection rule**: highest delta_val → fewest parents → simpler `(c, rho)` → higher |alpha|
- **Plateau cutoff**: `delta_val >= 0.25pp` to bake

## Replay in viewer

These networks are loaded into the Brain Replay viewer at `docs/pages/brain_replay/`. Run from repo root:

```
python tools/export_manual_grow_to_viewer.py
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Then open `http://localhost:8765/pages/brain_replay/`.

## Findings

See the **2026-04-13** section of `docs/wiki/Timeline-Archive.md` for the 12 published findings derived from these networks (label-polarity rule, threshold baseline showdown, balanced-bit stall, int8 quant lossless, voting semantics fix).

## Architectural note

This is **not** the single-network grower roadmap. It is a parallel exploration of the c19 *activation* via 72 small ensembles. The single-network roadmap is `instnct-core/examples/neuron_grower.rs` writing to `instnct-core/results/neuron_grower_persistent/`.
