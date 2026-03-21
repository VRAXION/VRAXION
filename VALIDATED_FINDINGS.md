# VRAXION Validated Findings

This page is the **canonical public evidence summary** for the repo.

It exists so the strongest current findings do not live only in issue traffic.

Repo-tracked docs are canonical. The GitHub wiki is treated as a mirrored secondary surface.

## What Matters Most Right Now

- **Current mainline:** [`v4.2/model/graph.py`](v4.2/model/graph.py) still ships `THRESHOLD = 0.5`, `INJ_SCALE = 3.0`, `DRIVE = 0.6`, plus co-evolved per-neuron `theta` / `decay` and nonnegative charge dynamics.
- **Current recipe candidate on `main`:** [`v4.2/english_1024n_18w.py`](v4.2/english_1024n_18w.py) now uses `8` ticks with a triangle-derived `2 add / 1 flip / 5 decay` schedule.
- **Strongest schedule result so far:** voltage medium leak reached `22.11%` peak / `21.46%` plateau.
- **Best compact learnable control policy so far:** the 3-angle decision-tree schedule reached `20.05%` at `156` edges.

## How To Read This Page

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: experimentally supported, but not yet promoted into the canonical code path.
- **Experimental branch**: active target or design direction, not yet validated as the default.

## Current Mainline

The canonical code path for `main` is [`v4.2/model/graph.py`](v4.2/model/graph.py).

Current mainline constants in that file:

- `THRESHOLD = 0.5`
- `INJ_SCALE = 3.0`
- `DRIVE = 0.6`
- per-neuron `theta` and `decay` are part of the mainline implementation
- charge uses nonnegative ReLU-style dynamics in the forward pass

Anything that differs from those settings should be described as a **Validated finding** or **Experimental branch**, not as the live default.

The current English recipe candidate on `main` is [`v4.2/english_1024n_18w.py`](v4.2/english_1024n_18w.py). It is useful evidence, but it is not the canonical architecture default.

## Evidence Table

| Topic | Status | Result | Code status |
|---|---|---|---|
| Charge ReLU ([66ce511](https://github.com/VRAXION/VRAXION/commit/66ce511d58b71cecbd92adc04f307299b3fc414b)) | Current mainline | Replacing symmetric clip with nonnegative charge unlocked flip accepts and improved English eval by about `+6%` | Promoted into `graph.py` forward paths |
| Flip mutation ([#112](https://github.com/VRAXION/VRAXION/issues/112)) | Validated finding | `flip` beat float weight perturbation by `+1.89%` on English 1024n | Not promoted into `graph.py` defaults; candidate for the mixed swarm implementation |
| Low theta + `INJ_SCALE=1.0` ([#113](https://github.com/VRAXION/VRAXION/issues/113)) | Validated finding | `scale=1.0 + theta=0.03` beat `scale=3.0 + theta=0.1` (`12.91%` vs `11.01%`) | Not promoted into `graph.py` defaults |
| 8 ticks + decay slot candidate ([2b4de88](https://github.com/VRAXION/VRAXION/commit/2b4de887656d5061a944d7f85b0bb2a875f767e4), [36086a0](https://github.com/VRAXION/VRAXION/commit/36086a0a58b02dad3413f883fdfd7d153108ed66)) | Validated finding | `8` ticks beat `6`, the decay-slot schedule reached `19.95%`, and the winning recipe was promoted into `english_1024n_18w.py` | Promoted into the current English recipe candidate on `main`, not into `graph.py` defaults |
| Decay resample mutation ([a5419e2](https://github.com/VRAXION/VRAXION/commit/a5419e22795af522afa2e2d8e292dd495f6c909f)) | Validated finding | Single-neuron resample in `[0.01, 0.5]` beat local decay perturbation and produced differentiated decay rates `[0.081-0.235]` | Not promoted into the current recipe or `graph.py` defaults |
| Voltage medium leak schedule ([b971613](https://github.com/VRAXION/VRAXION/commit/b971613550d881a7298690a2016339486e4c8244)) | Validated finding | Strongest schedule result so far: `22.11%` peak / `21.46%` plateau | Not promoted into the current recipe or `graph.py` defaults |
| Decision-tree schedule ([f7e6185](https://github.com/VRAXION/VRAXION/commit/f7e618511217d9b2905d93b30d7523a0be1fd79d)) | Validated finding | `20.05%` at `156` edges, with the best edge quality among the learnable schedule policies tested | Not promoted into the current recipe or `graph.py` defaults |
| Triangle-derived candidate schedule ([fdae6d6](https://github.com/VRAXION/VRAXION/commit/fdae6d6cd79a3554f23fbe94ab5412cea3e216d1)) | Validated finding | The triangle convergence result was distilled into a fixed `2 add / 1 flip / 5 decay` schedule for the current English recipe candidate on `main` | Promoted into `english_1024n_18w.py`, not into `graph.py` defaults |
| Mixed 18-worker swarm ([#114](https://github.com/VRAXION/VRAXION/issues/114)) | Experimental branch | Current next build target for English training | Not part of the canonical mainline yet |

## Historical Context

The earlier `Proven Findings` page belonged to the Diamond Code / pre-INSTNCT era. It is no longer the active public evidence surface. Current evidence lives here, in `Validated Findings`.

- External memory was already established as load-bearing in the older architecture line.
- Routing-first / content-second learning appeared as a recurring pattern before the current INSTNCT work.
- Task difficulty had to match actual architectural capability; impossible tasks created false bottleneck diagnoses.
- Processing depth mattered more than simply widening the local context window.

Those historical findings are useful context, not current mainline claims.

## How To Read This Repo

- If a result is in an issue but not in the canonical code path, treat it as a **Validated finding** or **Experimental branch**.
- If a page claims a setting is a live mainline default, that claim should be verifiable in [`v4.2/model/graph.py`](v4.2/model/graph.py).
- If code and narrative disagree, the code wins for “Current mainline,” and the narrative should be fixed.
