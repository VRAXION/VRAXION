# VRAXION

VRAXION is building **INSTNCT / SWG v4.2**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of running backpropagation through a fixed topology.

This repository is meant to be a credible front door for technical buyers and engineers. It should let a first-time reader answer five things quickly:

1. what VRAXION is,
2. why the architecture is different,
3. what is actually proven,
4. what the current canonical code path is,
5. how to verify one claim in minutes.

## Why This Architecture Is Different

INSTNCT / SWG v4.2 is built around a small set of unusual choices:

- **Passive I/O**: `W_in` and `W_out` are fixed random projections, not learned layers.
- **Self-wiring core**: the only learnable structure is a hidden-to-hidden ternary graph.
- **Persistent internal state**: neurons keep charge and state across ticks instead of acting as one-shot activations.
- **Mutation + selection**: training is done by graph edits and acceptance tests, not gradient descent through the graph.

The canonical reference implementation is [`v4.2/model/graph.py`](v4.2/model/graph.py).

## Status Taxonomy

To keep the public story truthful, this repo uses three labels consistently:

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: a result backed by a concrete experiment, but not yet promoted into the canonical code path.
- **Experimental branch**: an active build target or design direction that is not yet a validated default.

If code and docs disagree, **code wins for “Current mainline.”**

The repo-tracked docs are the canonical public source. The GitHub wiki is a secondary mirror, not an independent source of truth.

## Current State

### Current mainline

- The live canonical path is [`v4.2/model/graph.py`](v4.2/model/graph.py).
- The stable reference is the NumPy self-wiring graph with passive I/O, a ternary hidden mask, and persistent charge/state dynamics.
- Recent English sweeps around low-theta training and signal scaling are **not** described here as baked defaults until they land in that code path.

### Validated findings

- **Flip mutation** is the strongest structural mutation found so far on English 1024n; float weight perturbation lost badly ([#112](https://github.com/VRAXION/VRAXION/issues/112)).
- **`INJ_SCALE=1.0` + low theta** beat the older `scale=3.0` hack in empty-start English sweeps, but that result is still tracked as a validated finding rather than a shipped default ([#113](https://github.com/VRAXION/VRAXION/issues/113)).

The canonical evidence summary lives in [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md).

### Experimental branch

- The current next candidate is the **mixed 18-worker swarm** for English training ([#114](https://github.com/VRAXION/VRAXION/issues/114)).
- It is an active implementation target, not the live training path on `main`.

## 5-Minute Proof

Create an environment, install the minimal dependencies, and run the same checks used to keep the public repo honest:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m compileall v4.2 tools
python v4.2/tests/test_model.py
python tools/check_public_surface.py
```

These commands verify:

- the reference code compiles,
- the reference self-wiring model passes its stress test,
- the public-facing docs still agree with the canonical code path.

## Read Next

- [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md) — canonical evidence summary
- [`v4.2/README.md`](v4.2/README.md) — architecture-line map and technical entry points
- [VRAXION architecture page (SWG v4.2)](https://github.com/VRAXION/VRAXION/wiki/SWG-v4.2-Architecture)
- [Issue #114](https://github.com/VRAXION/VRAXION/issues/114) — current next build target

## License

- Noncommercial: [LICENSE](LICENSE)
- Commercial terms: [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)
- Citation: [CITATION.cff](CITATION.cff)
