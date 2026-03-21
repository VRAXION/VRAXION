<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# SWG v4.2 Architecture

VRAXION is building **INSTNCT / SWG v4.2**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of running backpropagation through a fixed topology.

This page describes the current architecture line. The wiki is a **mirrored secondary surface**, and code on `main` is the source of truth for **Current mainline**. Use **Validated finding** for experiment-backed results that are not yet shipped, and **Experimental branch** for active targets that are not live defaults.

## What This Architecture Is

INSTNCT / SWG v4.2 treats the learnable object as structure rather than a dense weight stack. Fixed passive I/O projections feed a self-wiring hidden graph whose connectivity is changed by mutation + selection, while neurons keep persistent internal state across ticks.

That changes what is being optimized: the model is not primarily learning a layer stack with backpropagation, but a directed ternary graph and its recurrent state dynamics.

## Architecture In One Screen

```text
input -> W_in -> hidden ternary graph -> W_out -> output
              persistent charge/state across ticks
```

- `W_in` and `W_out` are passive I/O projections.
- The hidden-to-hidden graph is ternary and self-wiring.
- Charge/state persists across ticks.
- Training is mutation + selection, not backpropagation through the graph.

## What Is Fixed vs Learnable

| Component | Role |
|---|---|
| `W_in` | Fixed random projection |
| `W_out` | Fixed random projection |
| Hidden-to-hidden mask | Learnable ternary graph |
| Charge / state | Dynamic runtime state |

## Current Mainline

- **Canonical code path:** [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)
- **Shipped constants on `main`:**
  - `THRESHOLD = 0.5`
  - `INJ_SCALE = 3.0`
  - `DRIVE = 0.6`

Anything else should be treated as a **Validated finding** or **Experimental branch** until it is promoted into `graph.py`.

## Validated Findings Around This Architecture

| Topic | Label | Outcome | In `main`? | Supporting link |
|---|---|---|---|---|
| `flip` mutation | Validated finding | Beat float weight perturbation by `+1.89%` eval on English next-byte training | No | [#112](https://github.com/VRAXION/VRAXION/issues/112) |
| `scale=1.0 + low theta` | Validated finding | Beat the older `scale=3.0 + theta=0.1` recipe in empty-start English sweeps | No | [#113](https://github.com/VRAXION/VRAXION/issues/113) |
| mixed 18-worker swarm | Experimental branch | Current next build target for English training | No | [#114](https://github.com/VRAXION/VRAXION/issues/114) |

The canonical evidence summary lives in [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md).

## How To Verify Quickly

These checks verify both reference-code health and public-truth alignment:

```bash
python -m compileall v4.2 tools
python v4.2/tests/test_model.py
python tools/check_public_surface.py
```

## Read Next

- [[Home]]
- [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md)
- [`README.md`](https://github.com/VRAXION/VRAXION/blob/main/README.md)

If the GitHub wiki render looks incomplete, use [Pages](https://vraxion.github.io/VRAXION/) or the repo [`README.md`](https://github.com/VRAXION/VRAXION/blob/main/README.md).
