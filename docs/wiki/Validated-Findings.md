<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Validated Findings

This page is the public evidence summary for VRAXION.

The wiki is a **mirrored secondary surface**. The canonical repo-tracked source is [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md).

## What Matters Most Right Now

- **Current mainline:** [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py) still ships `THRESHOLD = 0.5`, `INJ_SCALE = 3.0`, `DRIVE = 0.6`, plus per-neuron `theta` / `decay` and nonnegative charge dynamics.
- **Current recipe candidate on `main`:** [`v4.2/english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/english_1024n_18w.py) now uses `8` ticks and an `add/add/add/flip/theta/decay` schedule.
- **Strongest schedule result so far:** voltage medium leak reached `22.11%` peak / `21.46%` plateau.
- **Best compact learnable control policy so far:** the 3-angle decision-tree schedule reached `20.05%` at `156` edges.

## How To Read This Page

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: experimentally supported, but not yet promoted into the canonical code path.
- **Experimental branch**: active target or design direction, not yet validated as the default.

If code and docs disagree, **code wins for Current mainline**.

## Evidence Table

| Topic | Label | Result | Code status |
|---|---|---|---|
| Charge ReLU ([66ce511](https://github.com/VRAXION/VRAXION/commit/66ce511d58b71cecbd92adc04f307299b3fc414b)) | Current mainline | Replacing symmetric clip with nonnegative charge unlocked flip accepts and improved English eval by about `+6%` | Promoted into `graph.py` forward paths |
| Flip mutation ([#112](https://github.com/VRAXION/VRAXION/issues/112)) | Validated finding | `flip` beat float weight perturbation by `+1.89%` on English 1024n | Not promoted into `graph.py` defaults |
| Low theta + `INJ_SCALE=1.0` ([#113](https://github.com/VRAXION/VRAXION/issues/113)) | Validated finding | `scale=1.0 + theta=0.03` beat `scale=3.0 + theta=0.1` (`12.91%` vs `11.01%`) | Not promoted into `graph.py` defaults |
| 8 ticks + decay slot candidate ([2b4de88](https://github.com/VRAXION/VRAXION/commit/2b4de887656d5061a944d7f85b0bb2a875f767e4), [36086a0](https://github.com/VRAXION/VRAXION/commit/36086a0a58b02dad3413f883fdfd7d153108ed66)) | Validated finding | `8` ticks beat `6`, the decay-slot schedule reached `19.95%`, and the winning recipe was promoted into `english_1024n_18w.py` | Promoted into the current English recipe candidate on `main`, not into `graph.py` defaults |
| Decay resample mutation ([a5419e2](https://github.com/VRAXION/VRAXION/commit/a5419e22795af522afa2e2d8e292dd495f6c909f)) | Validated finding | Single-neuron resample in `[0.01, 0.5]` beat local decay perturbation and produced differentiated decay rates `[0.081-0.235]` | Not promoted into the current recipe or `graph.py` defaults |
| Voltage medium leak schedule ([b971613](https://github.com/VRAXION/VRAXION/commit/b971613550d881a7298690a2016339486e4c8244)) | Validated finding | Strongest schedule result so far: `22.11%` peak / `21.46%` plateau | Not promoted into the current recipe or `graph.py` defaults |
| Decision-tree schedule ([f7e6185](https://github.com/VRAXION/VRAXION/commit/f7e618511217d9b2905d93b30d7523a0be1fd79d)) | Validated finding | `20.05%` at `156` edges, with the best edge quality among the learnable schedule policies tested | Not promoted into the current recipe or `graph.py` defaults |
| Mixed 18-worker swarm ([#114](https://github.com/VRAXION/VRAXION/issues/114)) | Experimental branch | Current next build target for English training | Not part of the canonical mainline yet |

## Historical Context

The earlier `Proven Findings` page belonged to the Diamond Code / pre-INSTNCT era. It is no longer the active public evidence surface. Current evidence lives here, in `Validated Findings`.

- External memory was already established as load-bearing in the older architecture line.
- Routing-first / content-second learning appeared as a recurring pattern before the current INSTNCT work.
- Task difficulty had to match actual architectural capability; impossible tasks created false bottleneck diagnoses.
- Processing depth mattered more than simply widening the local context window.

Those historical findings are useful context, not current mainline claims.

## Read Next

- [Home](Home)
- [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture)
- [Engineering Protocol](Engineering)
- [Project Timeline](Release-Notes)
- [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md)

If the GitHub wiki render looks incomplete, use [Pages](https://vraxion.github.io/VRAXION/) or the repo [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md).
