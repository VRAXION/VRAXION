# VRAXION Validated Findings

This page is the **canonical public evidence summary** for the repo.

It exists so the strongest current findings do not live only in issue traffic.

Repo-tracked docs are canonical. The GitHub wiki is treated as a mirrored secondary surface.

## Status Taxonomy

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

## Evidence Snapshot

| Topic | Status | Exact setup | Outcome | Mainline status |
|---|---|---|---|---|
| Charge ReLU ([66ce511](https://github.com/VRAXION/VRAXION/commit/66ce511d58b71cecbd92adc04f307299b3fc414b)) | Current mainline | English 1024n charge-activation sweep, 200 steps | Replacing symmetric clip with nonnegative charge unlocked flip accepts and improved eval by about `+6%` | Promoted into `graph.py` forward paths |
| Flip mutation ([#112](https://github.com/VRAXION/VRAXION/issues/112)) | Validated finding | English next-byte, 1024 neurons, 18 workers, 200 steps | `flip` beat float weight perturbation by `+1.89%` eval | Not promoted into `graph.py` defaults; candidate for the mixed swarm implementation |
| Low theta + `INJ_SCALE=1.0` ([#113](https://github.com/VRAXION/VRAXION/issues/113)) | Validated finding | Empty-start English sweep, 18 workers, 100 steps | `scale=1.0 + theta=0.03` beat `scale=3.0 + theta=0.1` (`12.91%` vs `11.01%`) | Not promoted into `graph.py` defaults yet |
| 8 ticks + decay slot candidate ([2b4de88](https://github.com/VRAXION/VRAXION/commit/2b4de887656d5061a944d7f85b0bb2a875f767e4), [36086a0](https://github.com/VRAXION/VRAXION/commit/36086a0a58b02dad3413f883fdfd7d153108ed66)) | Validated finding | Tick sweep (200 steps) plus fixed/learnable schedule sweeps on English 1024n | `8` ticks beat `6`, a decay-dominant schedule reached `19.95%`, and the winning recipe was promoted into `english_1024n_18w.py` on `main` | Promoted into the current English recipe candidate on `main`, not into `graph.py` defaults |
| Voltage medium leak schedule ([b971613](https://github.com/VRAXION/VRAXION/commit/b971613550d881a7298690a2016339486e4c8244)) | Validated finding | 8 ticks, bigram 2-seq eval, charge ReLU, plateau sweep | Strongest schedule result so far: `22.11%` peak / `21.46%` plateau | Not promoted into the current recipe or `graph.py` defaults |
| Decision-tree schedule ([f7e6185](https://github.com/VRAXION/VRAXION/commit/f7e618511217d9b2905d93b30d7523a0be1fd79d)) | Validated finding | 3 learnable angles, `cos^2` split tree, 8 ticks, bigram 2-seq eval | `20.05%` at `156` edges, with the best edge-quality/efficiency among the learnable schedule policies tested | Not promoted into the current recipe or `graph.py` defaults |
| Mixed 18-worker swarm ([#114](https://github.com/VRAXION/VRAXION/issues/114)) | Experimental branch | Heterogeneous worker schedule for English training | Current next build target | Not part of the canonical mainline yet |

## How To Read This Repo

- If a result is in an issue but not in the canonical code path, treat it as a **Validated finding** or **Experimental branch**.
- If a page claims a setting is a live mainline default, that claim should be verifiable in [`v4.2/model/graph.py`](v4.2/model/graph.py).
- If code and narrative disagree, the code wins for “Current mainline,” and the narrative should be fixed.
