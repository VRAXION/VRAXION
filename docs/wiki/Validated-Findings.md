<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Validated Findings

This page is the public evidence board for VRAXION. Use it for the strongest reproducible results, whether they are already **Current mainline**, still a **Validated finding**, or still only an **Experimental branch**.

## Best Current Evidence

- **Current mainline reference point:** [`instnct/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/instnct/model/graph.py) still ships `THRESHOLD = 0.5`, `INJ_SCALE = 3.0`, `DRIVE = 0.6`, plus per-neuron `theta` / `decay` and nonnegative charge dynamics.
- **Current recipe candidate on `main`:** [`instnct/recipes/english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/instnct/recipes/english_1024n_18w.py) now uses `8` ticks with a triangle-derived `2 add / 1 flip / 5 decay` schedule; it still uses the existing float signed edge mask.
- **Strongest schedule result so far:** voltage medium leak reached `22.11%` peak / `21.46%` plateau.
- **Best compact learnable control policy so far:** the 3-angle decision-tree schedule reached `20.05%` at `156` edges.
- **Best edge-representation quality result so far:** sign+mag + magnitude resample reached `18.69%` at `155` edges (`q=0.121`), but it is not promoted into the current recipe candidate or `graph.py` defaults.
- **Strongest current task-learning input result so far:** window=2 superposition reached `21.8%`, beating the `w=1` baseline by `+72%`.
- **Strongest current task-memory evaluation result so far:** word-pair log-likelihood eval reached `23.8%`, beating bigram cosine at `18.8%`.

## Use This Page When

- you want the strongest evidence first
- you want to know which results are shipped, reproducible-but-unpromoted, or still active targets
- you need one page instead of issue archaeology

## Evidence Table

| Topic | Label | Result | Code status |
|---|---|---|---|
| Charge ReLU ([66ce511](https://github.com/VRAXION/VRAXION/commit/66ce511d58b71cecbd92adc04f307299b3fc414b)) | Current mainline | Replacing symmetric clip with nonnegative charge unlocked flip accepts and improved English eval by about `+6%` | Promoted into `graph.py` forward paths |
| Flip mutation ([#112](https://github.com/VRAXION/VRAXION/issues/112)) | Validated finding | `flip` beat float weight perturbation by `+1.89%` on English 1024n | Not promoted into `graph.py` defaults |
| Low theta + `INJ_SCALE=1.0` ([#113](https://github.com/VRAXION/VRAXION/issues/113)) | Validated finding | `scale=1.0 + theta=0.03` beat `scale=3.0 + theta=0.1` (`12.91%` vs `11.01%`) | Not promoted into `graph.py` defaults |
| 8 ticks + current English candidate schedule ([2b4de88](https://github.com/VRAXION/VRAXION/commit/2b4de887656d5061a944d7f85b0bb2a875f767e4), [36086a0](https://github.com/VRAXION/VRAXION/commit/36086a0a58b02dad3413f883fdfd7d153108ed66), [fdae6d6](https://github.com/VRAXION/VRAXION/commit/fdae6d6cd79a3554f23fbe94ab5412cea3e216d1)) | Validated finding | `8` ticks beat `6`, and the current candidate on `main` now uses the triangle-derived `2 add / 1 flip / 5 decay` schedule | Promoted into the current English recipe candidate on `main`, not into `graph.py` defaults |
| Decay resample mutation ([a5419e2](https://github.com/VRAXION/VRAXION/commit/a5419e22795af522afa2e2d8e292dd495f6c909f)) | Validated finding | Single-neuron resample in `[0.01, 0.5]` beat local decay perturbation and produced differentiated decay rates `[0.081-0.235]` | Not promoted into the current recipe or `graph.py` defaults |
| Voltage medium leak schedule ([b971613](https://github.com/VRAXION/VRAXION/commit/b971613550d881a7298690a2016339486e4c8244)) | Validated finding | Strongest schedule result so far: `22.11%` peak / `21.46%` plateau | Not promoted into the current recipe or `graph.py` defaults |
| Decision-tree schedule ([f7e6185](https://github.com/VRAXION/VRAXION/commit/f7e618511217d9b2905d93b30d7523a0be1fd79d)) | Validated finding | `20.05%` at `156` edges, with the best edge quality among the learnable schedule policies tested | Not promoted into the current recipe or `graph.py` defaults |
| Sign+mag + magnitude resample ([41f3622](https://github.com/VRAXION/VRAXION/commit/41f3622e654a79ffba0c95421b5e8a5c0f354364)) | Validated finding | Bool sign + uint8 magnitude with full magnitude resample reached `18.69%` at `155` edges (`q=0.121`), beating sign+mag free and delivering the best quality-per-edge result in the edge-format sweep without taking the best raw accuracy overall | Not promoted into the current recipe candidate or `graph.py` defaults |
| Window=2 input superposition ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)) | Validated finding | `w=2` reached `21.8%`, beating `w=1` at `12.7%` and all wider tested windows on the current task-learning sweep | Not promoted into the current recipe or `graph.py` defaults |
| Word-pair log-likelihood eval ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)) | Validated finding | `23.8%` on short associative-memory probes, beating bigram cosine at `18.8%` | Not part of the canonical mainline yet |
| Context-dependent task learning ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)) | Experimental branch | Current next build target: input-window injection, word-pair memory, and stronger evaluation for nontrivial tasks | Not part of the canonical mainline yet |

<details>
<summary>Historical Context</summary>

The earlier `Proven Findings` page belonged to the Diamond Code / pre-INSTNCT era. It is no longer the active public evidence surface. Current evidence lives here, in `Validated Findings`.

- External memory was already established as load-bearing in the older architecture line.
- Routing-first / content-second learning appeared as a recurring pattern before the current INSTNCT work.
- Task difficulty had to match actual architectural capability; impossible tasks created false bottleneck diagnoses.
- Processing depth mattered more than simply widening the local context window.

Those historical findings are useful context, not current mainline claims.

</details>

## Read Next

- [VRAXION Home](Home)
- [INSTNCT Architecture](INSTNCT-Architecture)
- [Engineering Protocol](Engineering)
- [Project Timeline](Release-Notes)
- [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md)
