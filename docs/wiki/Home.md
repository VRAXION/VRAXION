<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# VRAXION Wiki

VRAXION is building **INSTNCT**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of running backpropagation through a fixed topology.

This wiki is a **mirrored secondary surface**. The primary public entry points are the repo [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md) and [Pages](https://vraxion.github.io/VRAXION/).

## Status Taxonomy

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: an experiment-backed result that has not been promoted into the canonical code path yet.
- **Experimental branch**: an active build target or design direction, not a live default.

If code and docs disagree, **code wins for Current mainline**.

## What VRAXION Is

VRAXION is both a company and an engineering effort focused on a new architecture line:

- fixed random passive I/O projections,
- a self-wiring ternary hidden graph,
- persistent internal state across ticks,
- mutation + selection training instead of backpropagation through the graph.

The current architecture page is [[SWG-v4.2-Architecture|VRAXION Architecture (INSTNCT)]].

## Why It Matters

The aim is not “another model wrapper.” The aim is to make the learnable object itself structural: a directed graph that changes its own wiring while keeping the reference implementation inspectable and reproducible.

For technical buyers, the important question is therefore not just benchmark score. It is whether the architecture, proof surface, and canonical code path all tell the same story.

## Current Mainline

- Canonical code path: [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)
- Architecture line: `INSTNCT`
- Canonical public evidence summary: [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md)
- Mainline implementation includes per-neuron `theta` / `decay` and nonnegative charge dynamics.

Anything not actually shipped in that code path must be labeled as a **Validated finding** or **Experimental branch**, not as a live default.

## What Is Already Validated

| Topic | Label | Mainline state |
|---|---|---|
| `flip` mutation | Validated finding | Not promoted into `graph.py` defaults |
| `scale=1.0 + low theta` | Validated finding | Not promoted into `graph.py` defaults |
| `8` ticks + decay-aware schedule | Validated finding | Not promoted into `graph.py` defaults |
| mixed 18-worker swarm | Experimental branch | Active target, not current mainline |

See [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md) for the canonical evidence summary, with issue links as supporting references rather than the front door.

## 5-Minute Proof

Run the same checks used to keep the public repo aligned with the code:

```bash
python -m compileall v4.2 tools
python v4.2/tests/test_model.py
python tools/check_public_surface.py
```

## Start Here

- [[SWG-v4.2-Architecture|VRAXION Architecture (INSTNCT)]] — architecture line and public status
- [`README.md`](https://github.com/VRAXION/VRAXION/blob/main/README.md) — repo front door
- [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md) — canonical evidence summary

If the GitHub wiki render looks incomplete or noisy, use [Pages](https://vraxion.github.io/VRAXION/) or the repo [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md).
