<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# VRAXION Wiki

VRAXION is building **INSTNCT / SWG v4.2**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of running backpropagation through a fixed topology.

This wiki is now organized around one public contract:

1. what VRAXION is,
2. why the architecture is different,
3. what is already proven,
4. what is actually shipped on `main`,
5. how to verify one claim quickly.

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

The current architecture page is [[SWG-v4.2-Architecture]].

## Why It Matters

The aim is not “another model wrapper.” The aim is to make the learnable object itself structural: a directed graph that changes its own wiring while keeping the reference implementation inspectable and reproducible.

For technical buyers, the important question is therefore not just benchmark score. It is whether the architecture, proof surface, and canonical code path all tell the same story.

## Current Mainline

- Canonical code path: [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)
- Architecture line: `INSTNCT / SWG v4.2`
- Canonical public evidence summary: [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md)

Anything not actually shipped in that code path must be labeled as a **Validated finding** or **Experimental branch**, not as a live default.

## What Is Already Validated

- **Flip mutation** beat float weight perturbation on English next-byte training.
- **`INJ_SCALE=1.0` + low theta** beat the older `scale=3.0` hack in an empty-start English sweep.
- **Mixed 18-worker swarm** is the current next active build target, but still experimental.

These outcomes are summarized in [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md) and tracked in [issue #112](https://github.com/VRAXION/VRAXION/issues/112), [issue #113](https://github.com/VRAXION/VRAXION/issues/113), and [issue #114](https://github.com/VRAXION/VRAXION/issues/114).

## 5-Minute Proof

Run the same checks used to keep the public repo aligned with the code:

```bash
python -m compileall v4.2 tools
python v4.2/tests/test_model.py
python tools/check_public_surface.py
```

## Start Here

- [[SWG-v4.2-Architecture]] — architecture line and public status
- [`README.md`](https://github.com/VRAXION/VRAXION/blob/main/README.md) — repo front door
- [`VALIDATED_FINDINGS.md`](https://github.com/VRAXION/VRAXION/blob/main/VALIDATED_FINDINGS.md) — canonical evidence summary
- [Issue #114](https://github.com/VRAXION/VRAXION/issues/114) — current next build target
