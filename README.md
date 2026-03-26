# VRAXION

VRAXION is building **INSTNCT**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of running backpropagation through a fixed topology.

> **Core thesis**
>
> **Inference emerges as the fixed point of destructive interference.**
>
> Signal enters a structured recurrent substrate, incompatible propagation paths cancel through destructive interference, and the surviving pattern — the fixed point — is read out as inference. This is a research thesis for the architecture line, not a separate claim of achieved sentience.

This repository is meant to be a credible front door for technical buyers and engineers. It should let a first-time reader answer five things quickly:

1. what VRAXION is,
2. why the architecture is different,
3. what is actually proven,
4. what the current canonical code path is,
5. how to verify one claim in minutes.

## Release Snapshot

- **Current canonical public release:** [`v4.2.0`](https://github.com/VRAXION/VRAXION/releases/tag/v4.2.0) (`stable`)
- **Next public milestone:** preparation toward `v5.0.0 Public Beta`
- **Internal code path:** [`instnct/`](instnct/) (repo path, not the public release label)
- **Current mainline code path:** [`instnct/model/graph.py`](instnct/model/graph.py)

## Why This Architecture Is Different

INSTNCT is built around a small set of unusual choices:

- **Passive I/O**: `input_projection` and `output_projection` are fixed random projections, not learned layers.
- **Self-wiring core**: the primary learnable structure is a hidden-to-hidden signed edge mask, with co-evolved per-neuron `theta` and `decay`.
- **Persistent internal state**: neurons keep charge and state across ticks instead of acting as one-shot activations.
- **Mutation + selection**: training is done by graph edits and acceptance tests, not gradient descent through the graph.

The canonical reference implementation is [`instnct/model/graph.py`](instnct/model/graph.py).

## Status Taxonomy

To keep the public story truthful, this repo uses three labels consistently:

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: a result backed by a concrete experiment, but not yet promoted into the canonical code path.
- **Experimental branch**: an active build target or design direction that is not yet a validated default.

If code and docs disagree, **code wins for “Current mainline.”**

The repo-tracked docs are the canonical public source. The GitHub wiki is a secondary mirror, not an independent source of truth.

Historical archive branch:

- The Diamond Code era is preserved on `archive/diamond-code-era-20260322`; it is not part of active `main`.
- The pre-beta research surface freeze is preserved on `archive/instnct-surface-freeze-20260322`; bulky raw experiment dumps and retired exploratory surfaces no longer live on `main`.

Retired line names and older local folders belong in [Project Timeline](https://github.com/VRAXION/VRAXION/wiki/Release-Notes), not in the current front-door stack.

## Current State

### Current mainline

- The live canonical path is [`instnct/model/graph.py`](instnct/model/graph.py).
- The stable reference is the NumPy self-wiring graph with passive I/O, a signed hidden edge mask, co-evolved per-neuron `theta` / `decay`, and nonnegative charge dynamics.
- The current English recipe candidate on `main` is [`instnct/recipes/train_english_1024n_18w.py`](instnct/recipes/train_english_1024n_18w.py); it currently uses an `8`-tick triangle-derived `2 add / 1 flip / 5 decay` schedule, but it is still a candidate training script, not the canonical architecture default.
- English is the only first-class public lane on `main`; task-memory and GPU remain secondary validation surfaces rather than equal front-door lanes.
- Recent English sweeps around low-theta training and signal scaling are **not** described here as baked defaults until they land in that code path.

### Evidence snapshot

- **Charge ReLU** is now part of the current mainline forward path; replacing symmetric clip with nonnegative charge unlocked flip accepts and materially improved English training ([66ce511](https://github.com/VRAXION/VRAXION/commit/66ce511d58b71cecbd92adc04f307299b3fc414b)).
- **Flip mutation** is the strongest structural mutation found so far on English 1024n; float weight perturbation lost badly ([#112](https://github.com/VRAXION/VRAXION/issues/112)).
- **`INJ_SCALE=1.0` + low theta** beat the older `scale=3.0` hack in empty-start English sweeps, but that result is still tracked as a validated finding rather than a shipped default ([#113](https://github.com/VRAXION/VRAXION/issues/113)).
- **The current English recipe candidate** on `main` uses an `8`-tick triangle-derived `2 add / 1 flip / 5 decay` schedule, but that recipe is still not promoted into the canonical `graph.py` defaults ([fdae6d6](https://github.com/VRAXION/VRAXION/commit/fdae6d6cd79a3554f23fbe94ab5412cea3e216d1)).
- **Sign+mag + magnitude resample** is the strongest edge-representation quality result so far: `18.69%` at `155` edges (`q=0.121`). It beat sign+mag free and delivered the best quality-per-edge result in that sweep, but it is still a validated finding rather than the live recipe candidate ([41f3622](https://github.com/VRAXION/VRAXION/commit/41f3622e654a79ffba0c95421b5e8a5c0f354364)).
- **Decay resample** beat local perturbation for per-neuron decay tuning; resampling one neuron into `[0.01, 0.5]` reached `19.35%` and produced differentiated decay rates instead of leaving decay stuck at `0.15` ([a5419e2](https://github.com/VRAXION/VRAXION/commit/a5419e22795af522afa2e2d8e292dd495f6c909f)).
- **Schedule-control experiments** now have two validated lines: a higher-accuracy voltage/leak recipe (`22.11%` peak / `21.46%` plateau) and a simpler 3-angle decision tree (`20.05%` at `156` edges with better edge quality) ([b971613](https://github.com/VRAXION/VRAXION/commit/b971613550d881a7298690a2016339486e4c8244), [f7e6185](https://github.com/VRAXION/VRAXION/commit/f7e618511217d9b2905d93b30d7523a0be1fd79d)).
- **Window=2 input superposition** is the strongest current task-learning injection result so far: `21.8%` vs `12.7%` at `w=1`, making it the current leading context-window line for nontrivial task probes ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)).
- **Word-pair log-likelihood eval** beat bigram cosine on associative-memory probes (`23.8%` vs `18.8%`), showing that context-dependent tasks need a stronger scoring path than pure bigram cosine ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)).

Raw run dumps, archived sweeps, and older exploratory surfaces now live on `archive/instnct-surface-freeze-20260322`, not on active `main`.

The canonical evidence summary lives in [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md).

### Experimental branch

- The current next candidate is **context-dependent task learning**, especially windowed input injection, word-pair memory, and stronger evaluation for nontrivial tasks ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)).
- It is an active experimental target, not the live training path on `main`.

## 5-Minute Proof

Create an environment, install the minimal dependencies, and run the same checks used to keep the public repo honest:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m compileall instnct tools
python instnct/tests/test_model.py
python tools/check_public_surface.py
```

These commands verify:

- the reference code compiles,
- the reference self-wiring model passes its stress test,
- the public-facing docs still agree with the canonical code path.

## Read Next

- [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md) — canonical evidence summary
- [`instnct/README.md`](instnct/README.md) — architecture-line map and technical entry points
- [VRAXION architecture page (INSTNCT)](https://github.com/VRAXION/VRAXION/wiki/INSTNCT-Architecture)
- [Issue #114](https://github.com/VRAXION/VRAXION/issues/114) — current next build target

## License

- Noncommercial: [LICENSE](LICENSE)
- Commercial terms: [legal/LEGAL.md](legal/LEGAL.md)
- Brand and trademark notice: [legal/LEGAL.md](legal/LEGAL.md)
- Citation: [CITATION.cff](CITATION.cff)

The software license does not grant rights to use the **VRAXION** or **INSTNCT** names, logos, or brand assets except as described in [legal/LEGAL.md](legal/LEGAL.md).
