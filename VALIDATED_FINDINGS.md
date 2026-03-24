# VRAXION Validated Findings

This page is the **canonical public evidence summary** for the repo.

It exists so the strongest current findings do not live only in issue traffic.

Repo-tracked docs are canonical. The GitHub wiki is treated as a mirrored secondary surface.

## What Matters Most Right Now

- **Current mainline:** [`instnct/model/graph.py`](instnct/model/graph.py) now ships explicit per-instance defaults `DEFAULT_THETA = 0.1`, `DEFAULT_PROJECTION_SCALE = 3.0`, `DEFAULT_EDGE_MAGNITUDE = 1.0`, plus co-evolved per-neuron `theta` / `decay` and nonnegative charge dynamics.
- **Current recipe candidate on `main`:** [`instnct/recipes/english_1024n_18w.py`](instnct/recipes/english_1024n_18w.py) now uses `8` ticks with a triangle-derived `2 add / 1 flip / 5 decay` schedule; it still uses the existing float signed edge mask.
- **Strongest schedule result so far:** voltage medium leak reached `22.11%` peak / `21.46%` plateau.
- **Best compact learnable control policy so far:** the 3-angle decision-tree schedule reached `20.05%` at `156` edges.
- **Best edge-representation quality result so far:** sign+mag + magnitude resample reached `18.69%` at `155` edges (`q=0.121`), but it is not promoted into the current recipe candidate or `graph.py` defaults.
- **Strongest current task-learning input result so far:** window=2 superposition reached `21.8%`, beating the `w=1` baseline by `+72%`.
- **Strongest current task-memory evaluation result so far:** word-pair log-likelihood eval reached `23.8%`, beating bigram cosine at `18.8%`.
- **Binary Mask Finding:** Ternary masks (`{-1, 0, +1}`) are redundant; pure binary masks (`{0, 1}`) achieve equal accuracy (`86.5%`) because inhibition is implicitly handled by the input projection's negative values and per-neuron decay. This enables a **multiply-free forward pass**, which is `1.5x` to `3.4x` faster on edge hardware.
- **Surface policy on `main`:** English remains the only first-class public lane; task-memory and GPU remain tracked only as secondary validation surfaces.

## How To Read This Page

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: experimentally supported, but not yet promoted into the canonical code path.
- **Experimental branch**: active target or design direction, not yet validated as the default.

## Current Mainline

The canonical code path for `main` is [`instnct/model/graph.py`](instnct/model/graph.py).

Current mainline defaults in that file:

- `DEFAULT_THETA = 0.1`
- `DEFAULT_PROJECTION_SCALE = 3.0`
- `DEFAULT_EDGE_MAGNITUDE = 1.0`
- per-neuron `theta` and `decay` are part of the mainline implementation
- charge uses nonnegative ReLU-style dynamics in the forward pass

Anything that differs from those settings should be described as a **Validated finding** or **Experimental branch**, not as the live default.

The current English recipe candidate on `main` is [`instnct/recipes/english_1024n_18w.py`](instnct/recipes/english_1024n_18w.py). It currently uses the triangle-derived `2 add / 1 flip / 5 decay` schedule with the older float signed edge mask. It is useful evidence, but it is not the canonical architecture default.

The current secondary validation recipe on `main` is [`instnct/recipes/train_wordpairs_ll.py`](instnct/recipes/train_wordpairs_ll.py). It remains important for task-memory evaluation, but it is not a second front-door default.

Raw experiment dumps, retired sweeps, and archived exploratory probes now live on `archive/instnct-surface-freeze-20260322`, not on active `main`.

## Evidence Table

| Topic | Status | Result | Code status |
|---|---|---|---|
| Charge ReLU ([66ce511](https://github.com/VRAXION/VRAXION/commit/66ce511d58b71cecbd92adc04f307299b3fc414b)) | Current mainline | Replacing symmetric clip with nonnegative charge unlocked flip accepts and improved English eval by about `+6%` | Promoted into `graph.py` forward paths |
| Flip mutation ([#112](https://github.com/VRAXION/VRAXION/issues/112)) | Validated finding | `flip` beat float weight perturbation by `+1.89%` on English 1024n | Not promoted into `graph.py` defaults; candidate for the mixed swarm implementation |
| Low theta + `projection_scale=1.0` ([#113](https://github.com/VRAXION/VRAXION/issues/113)) | Validated finding | `scale=1.0 + theta=0.03` beat `scale=3.0 + theta=0.1` (`12.91%` vs `11.01%`) | Not promoted into `graph.py` defaults |
| 8 ticks + current English candidate schedule ([2b4de88](https://github.com/VRAXION/VRAXION/commit/2b4de887656d5061a944d7f85b0bb2a875f767e4), [36086a0](https://github.com/VRAXION/VRAXION/commit/36086a0a58b02dad3413f883fdfd7d153108ed66), [fdae6d6](https://github.com/VRAXION/VRAXION/commit/fdae6d6cd79a3554f23fbe94ab5412cea3e216d1)) | Validated finding | `8` ticks beat `6`, and the current candidate on `main` now uses the triangle-derived `2 add / 1 flip / 5 decay` schedule | Promoted into the current English recipe candidate on `main`, not into `graph.py` defaults |
| Decay resample mutation ([a5419e2](https://github.com/VRAXION/VRAXION/commit/a5419e22795af522afa2e2d8e292dd495f6c909f)) | Validated finding | Single-neuron resample in `[0.01, 0.5]` beat local decay perturbation and produced differentiated decay rates `[0.081-0.235]` | Not promoted into the current recipe or `graph.py` defaults |
| Voltage medium leak schedule ([b971613](https://github.com/VRAXION/VRAXION/commit/b971613550d881a7298690a2016339486e4c8244)) | Validated finding | Strongest schedule result so far: `22.11%` peak / `21.46%` plateau | Not promoted into the current recipe or `graph.py` defaults |
| Decision-tree schedule ([f7e6185](https://github.com/VRAXION/VRAXION/commit/f7e618511217d9b2905d93b30d7523a0be1fd79d)) | Validated finding | `20.05%` at `156` edges, with the best edge quality among the learnable schedule policies tested | Not promoted into the current recipe or `graph.py` defaults |
| Sign+mag + magnitude resample ([41f3622](https://github.com/VRAXION/VRAXION/commit/41f3622e654a79ffba0c95421b5e8a5c0f354364)) | Validated finding | Bool sign + uint8 magnitude with full magnitude resample reached `18.69%` at `155` edges (`q=0.121`), beating sign+mag free and delivering the best quality-per-edge result in the edge-format sweep without taking the best raw accuracy overall | Not promoted into the current recipe candidate or `graph.py` defaults |
| Window=2 input superposition ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)) | Validated finding | `w=2` reached `21.8%`, beating `w=1` at `12.7%` and all wider tested windows on the current task-learning sweep | Not promoted into the current recipe or `graph.py` defaults |
| Word-pair log-likelihood eval ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)) | Validated finding | `23.8%` on short associative-memory probes, beating bigram cosine at `18.8%` | Not part of the canonical mainline yet |
| Binary Mask ([0f9eba0](https://github.com/VRAXION/VRAXION/commit/0f9eba0a340f1a94165d21096054817d23f79038)) | Validated finding | Binary `{0, 1}` mask matches ternary accuracy (`86.5%`) because inhibition is handled by the input projection. Enables multiply-free forward pass (`3.4x` faster). | Promoted to `graph.py` multiply-free path |
| Context-dependent task learning ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)) | Experimental branch | Current next build target: input-window injection, word-pair memory, and stronger evaluation for nontrivial tasks | Not part of the canonical mainline yet |

## Historical Context

The earlier `Proven Findings` page belonged to the Diamond Code / pre-INSTNCT era. It is no longer the active public evidence surface. Current evidence lives here, in `Validated Findings`.

- External memory was already established as load-bearing in the older architecture line.
- Routing-first / content-second learning appeared as a recurring pattern before the current INSTNCT work.
- Task difficulty had to match actual architectural capability; impossible tasks created false bottleneck diagnoses.
- Processing depth mattered more than simply widening the local context window.

Those historical findings are useful context, not current mainline claims.

## How To Read This Repo

- If a result is in an issue but not in the canonical code path, treat it as a **Validated finding** or **Experimental branch**.
- If a page claims a setting is a live mainline default, that claim should be verifiable in [`instnct/model/graph.py`](instnct/model/graph.py).
- If code and narrative disagree, the code wins for “Current mainline,” and the narrative should be fixed.
