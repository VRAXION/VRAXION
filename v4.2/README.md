# INSTNCT

This directory contains the active architecture line behind VRAXION.

## Status Taxonomy

- **Current mainline**: what is actually shipped on `main`.
- **Validated finding**: experiment-backed result not yet promoted into the canonical code path.
- **Experimental branch**: active build target or prototype direction.

## Current Mainline

- Canonical reference: [`model/graph.py`](model/graph.py)
- Scaling/parity path: [`model/graph_v3.c`](model/graph_v3.c)
- Current English recipe candidate on `main`: [`english_1024n_18w.py`](english_1024n_18w.py) (`8` ticks, triangle-derived `2 add / 1 flip / 5 decay`)
- Core behavior:
  - fixed random passive I/O projections
  - signed hidden-to-hidden edge mask
  - co-evolved per-neuron `theta` and `decay`
  - nonnegative charge dynamics in the forward pass
  - persistent charge and state dynamics
  - mutation + selection training

This is the only code path that should be described as the live default.

## Validated Findings Not Yet Promoted

- [`flip` mutation](../VALIDATED_FINDINGS.md) is currently the strongest English structural mutation finding.
- [`scale=1.0 + theta=0.03`](../VALIDATED_FINDINGS.md) beat the older `INJ_SCALE=3.0` English setup in empty-start sweeps.
- The current English recipe candidate on `main` uses `8` ticks with a triangle-derived `2 add / 1 flip / 5 decay` schedule; see [Validated Findings](../VALIDATED_FINDINGS.md). That recipe is still not part of `model/graph.py`.
- [Sign+mag + magnitude resample](../VALIDATED_FINDINGS.md) reached `18.69%` at `155` edges (`q=0.121`) and became the best quality-per-edge result in the current edge-format sweep, but it is still not part of the live recipe candidate.
- [Decay resample for per-neuron tuning](../VALIDATED_FINDINGS.md) beat local perturbation and produced differentiated decay rates instead of leaving decay flat.
- [Voltage medium leak scheduling](../VALIDATED_FINDINGS.md) is the strongest current schedule finding by accuracy (`22.11%` peak / `21.46%` plateau).
- [The 3-angle decision-tree schedule](../VALIDATED_FINDINGS.md) is the strongest compact learnable control policy so far by edge quality.
- [Window=2 input superposition](../VALIDATED_FINDINGS.md) is the strongest current task-learning injection line so far at `21.8%`.
- [Word-pair log-likelihood evaluation](../VALIDATED_FINDINGS.md) beat bigram cosine on short associative-memory probes (`23.8%` vs `18.8%`).

These findings are important, but none of them should be described as a shipped default until `model/graph.py` actually adopts them.

## Experimental Next Target

- The current next build target is context-dependent task learning: windowed input injection, word-pair memory, and stronger evaluation for nontrivial tasks.
- It is an experimental direction, not the current canonical line.

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
