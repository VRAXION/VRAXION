<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# VRAXION Architecture

**Architecture line:** `INSTNCT`

VRAXION is building **INSTNCT**: a gradient-free self-wiring architecture that learns by changing its own graph instead of using backpropagation through a fixed layer stack.

This page explains the current architecture line in plain terms. The wiki is a **mirrored secondary surface**. Code on `main` is the source of truth for **Current mainline**. Results that are proven but not shipped are **Validated findings**. Active targets that are not shipped yet are **Experimental branches**.

## What This Architecture Is

Most neural systems learn by adjusting lots of weights inside a fixed topology. INSTNCT changes that. Here, the thing being learned is the hidden graph itself.

Input enters through fixed random projections, moves through a self-wiring hidden graph, and is read out through another fixed projection. The graph changes by mutation + selection, while neurons keep charge/state across ticks. In short: the model learns structure and state dynamics, not just layer weights.

## Architecture In One Screen

```text
input -> W_in -> hidden ternary graph -> W_out -> output
              persistent charge/state across ticks
```

- `W_in` and `W_out` are fixed random projections.
- The hidden graph is directed, ternary, and can rewire itself over time.
- Per-neuron `theta` and `decay` are co-evolved with the graph.
- Charge/state persists across ticks instead of resetting after one pass.
- Training happens by mutation + selection, not backpropagation through the graph.

## What Is Fixed vs Learnable

| Component | What it does |
|---|---|
| `W_in` | Fixed random projection |
| `W_out` | Fixed random projection |
| Hidden-to-hidden mask | Learnable graph structure |
| `theta` | Learnable per-neuron firing threshold |
| `decay` | Learnable per-neuron decay rate |
| Charge / state | Runtime state that changes while the model runs |

## Current Mainline

- **Canonical code path:** [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)
- **Current English recipe candidate on `main`:** [`v4.2/english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/english_1024n_18w.py)
- **What is actually shipped on `main`:**
  - `THRESHOLD = 0.5`
  - `INJ_SCALE = 3.0`
  - `DRIVE = 0.6`
- **Mainline runtime behavior:** per-neuron `theta` / `decay` and nonnegative charge dynamics

If a setting is not present in that file, it is not a live default. It should be read as a **Validated finding** or **Experimental branch** until it is promoted into `graph.py`.

## Validated Findings Around This Architecture

| Topic | Label | What it means | In `main`? | Supporting link |
|---|---|---|---|---|
| `flip` mutation | Validated finding | A better structural mutation than float weight perturbation on English next-byte training | No | [#112](https://github.com/VRAXION/VRAXION/issues/112) |
| `scale=1.0 + low theta` | Validated finding | A better experimental recipe than the older `scale=3.0 + theta=0.1` setup in empty-start English sweeps | No | [#113](https://github.com/VRAXION/VRAXION/issues/113) |
| `8` ticks + decay slot | Validated finding | The winning fixed schedule from the sweep line; now promoted into the current English recipe candidate on `main` | Not in `graph.py` | [36086a0](https://github.com/VRAXION/VRAXION/commit/36086a0a58b02dad3413f883fdfd7d153108ed66) |
| decay resample mutation | Validated finding | Full resample in `[0.01, 0.5]` beat local decay perturbation and created differentiated decay rates | No | [a5419e2](https://github.com/VRAXION/VRAXION/commit/a5419e22795af522afa2e2d8e292dd495f6c909f) |
| voltage medium leak schedule | Validated finding | The strongest schedule result so far by accuracy: `22.11%` peak / `21.46%` plateau | No | [b971613](https://github.com/VRAXION/VRAXION/commit/b971613550d881a7298690a2016339486e4c8244) |
| decision-tree schedule | Validated finding | A compact 3-angle learnable policy that reached `20.05%` with better edge quality than the voltage policy | No | [f7e6185](https://github.com/VRAXION/VRAXION/commit/f7e618511217d9b2905d93b30d7523a0be1fd79d) |
| mixed 18-worker swarm | Experimental branch | The current next build target for English training | No | [#114](https://github.com/VRAXION/VRAXION/issues/114) |

For the full evidence summary, use [Validated Findings](Validated-Findings).

## How To Verify Quickly

These checks verify both code health and public-truth alignment:

```bash
python -m compileall v4.2 tools
python v4.2/tests/test_model.py
python tools/check_public_surface.py
```

## Read Next

- [VRAXION Home](Home)
- [Validated Findings](Validated-Findings)
- [Engineering Protocol](Engineering)
- [`README.md`](https://github.com/VRAXION/VRAXION/blob/main/README.md)

If the GitHub wiki render looks incomplete, use [Pages](https://vraxion.github.io/VRAXION/) or the repo [`README.md`](https://github.com/VRAXION/VRAXION/blob/main/README.md).
