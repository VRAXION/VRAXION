# INSTNCT

This directory contains the active architecture line behind VRAXION.

## Status Taxonomy

- **Current mainline**: what is actually shipped on `main`.
- **Validated finding**: experiment-backed result not yet promoted into the canonical code path.
- **Experimental branch**: active build target or prototype direction.

## Current Mainline

- Canonical reference: [`model/graph.py`](model/graph.py)
- Scaling/parity path: [`model/graph_v3.c`](model/graph_v3.c)
- Core behavior:
  - fixed random passive I/O projections
  - ternary hidden-to-hidden mask
  - co-evolved per-neuron `theta` and `decay`
  - nonnegative charge dynamics in the forward pass
  - persistent charge and state dynamics
  - mutation + selection training

This is the only code path that should be described as the live default.

## Validated Findings Not Yet Promoted

- [`flip` mutation](../VALIDATED_FINDINGS.md) is currently the strongest English structural mutation finding.
- [`scale=1.0 + theta=0.03`](../VALIDATED_FINDINGS.md) beat the older `INJ_SCALE=3.0` English setup in empty-start sweeps.

Both are important, but neither should be described as a shipped default until `model/graph.py` actually adopts them.

## Experimental Next Target

- The current next build target is the mixed 18-worker swarm tracked in [issue #114](https://github.com/VRAXION/VRAXION/issues/114).
- It is a candidate training recipe, not the current canonical line.

## Recommended Entry Points

- [`model/graph.py`](model/graph.py) — canonical Python reference
- [`model/graph_v3.c`](model/graph_v3.c) — C scaling/parity path
- [`tests/test_model.py`](tests/test_model.py) — adversarial stress test
- [`tests/benchmark_ab.py`](tests/benchmark_ab.py) — benchmark comparison harness
- [`tests/gpu_experimental/`](tests/gpu_experimental/) — isolated GPU research branch notes and probes

## Quick Verification

```bash
python -m compileall v4.2
python v4.2/tests/test_model.py
python ../tools/check_public_surface.py
```
