<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# VRAXION Architecture

**Architecture line:** `INSTNCT / SWG v4.2`

VRAXION is building **INSTNCT / SWG v4.2**: a gradient-free self-wiring architecture that learns by changing its own graph instead of using backpropagation through a fixed layer stack.

This page explains the current architecture line in plain terms. The wiki is a **mirrored secondary surface**. Code on `main` is the source of truth for **Current mainline**. Results that are proven but not shipped are **Validated findings**. Active targets that are not shipped yet are **Experimental branches**.

## What This Architecture Is

Most neural systems learn by adjusting lots of weights inside a fixed topology. SWG v4.2 changes that. Here, the thing being learned is the hidden graph itself.

Input enters through fixed random projections, moves through a self-wiring hidden graph, and is read out through another fixed projection. The graph changes by mutation + selection, while neurons keep charge/state across ticks. In short: the model learns structure and state dynamics, not just layer weights.

## Architecture In One Screen

```text
input -> W_in -> hidden ternary graph -> W_out -> output
              persistent charge/state across ticks
```

- `W_in` and `W_out` are fixed random projections.
- The hidden graph is directed, ternary, and can rewire itself over time.
- Charge/state persists across ticks instead of resetting after one pass.
- Training happens by mutation + selection, not backpropagation through the graph.

## What Is Fixed vs Learnable

| Component | What it does |
|---|---|
| `W_in` | Fixed random projection |
| `W_out` | Fixed random projection |
| Hidden-to-hidden mask | Learnable graph structure |
| Charge / state | Runtime state that changes while the model runs |

## Current Mainline

- **Canonical code path:** [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)
- **What is actually shipped on `main`:**
  - `THRESHOLD = 0.5`
  - `INJ_SCALE = 3.0`
  - `DRIVE = 0.6`

If a setting is not present in that file, it is not a live default. It should be read as a **Validated finding** or **Experimental branch** until it is promoted into `graph.py`.

## Validated Findings Around This Architecture

| Topic | Label | What it means | In `main`? | Supporting link |
|---|---|---|---|---|
| `flip` mutation | Validated finding | A better structural mutation than float weight perturbation on English next-byte training | No | [#112](https://github.com/VRAXION/VRAXION/issues/112) |
| `scale=1.0 + low theta` | Validated finding | A better experimental recipe than the older `scale=3.0 + theta=0.1` setup in empty-start English sweeps | No | [#113](https://github.com/VRAXION/VRAXION/issues/113) |
| mixed 18-worker swarm | Experimental branch | The current next build target for English training | No | [#114](https://github.com/VRAXION/VRAXION/issues/114) |

For the full evidence summary, use [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md).

## How To Verify Quickly

These checks verify both code health and public-truth alignment:

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
