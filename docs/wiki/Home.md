<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# VRAXION Wiki

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-instnct-spiral.png" alt="INSTNCT spiral logo" width="360">
</p>

VRAXION is building **INSTNCT**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of running backpropagation through a fixed topology.

This is the lean technical reference for the public stack. Use [Pages](https://vraxion.github.io/VRAXION/) for the polished front door, the repo [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md) for the code-facing front door, and this page for the fastest orientation to what is current. Repo-tracked docs are canonical and this wiki is their mirrored secondary surface.

## At a Glance

- **Current mainline:** [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py) is the canonical shipped code path on `main`.
- **Best validated evidence right now:** voltage medium leak remains the strongest schedule result at `22.11%` peak / `21.46%` plateau.
- **Current next target:** mixed 18-worker swarm remains the main active build target.

## Use This Page When

- you need the current public stack in one screen
- you need the distinction between shipped code, validated evidence, and active experimental work
- you need the right next page without reconstructing it from issue traffic

Use this stack map to jump by question, not by page name.

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-public-stack-map.png" alt="VRAXION public stack hierarchy" width="740">
</p>

## What VRAXION Is

VRAXION is both a company and an engineering effort built around one architecture line: passive I/O projections, a self-wiring ternary hidden graph, persistent internal state across ticks, and mutation-selection training instead of backpropagation through the graph.

The current architecture page is [INSTNCT Architecture](SWG-v4.2-Architecture).

INSTNCT core anatomy at a glance:

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/instnct-at-a-glance-core.png" alt="INSTNCT core anatomy at a glance" width="740">
</p>

## Long-Horizon Mission

> In a stable loop, structure emerges.

VRAXION exists to advance machine consciousness as an engineering reality, and to ensure that the underlying mechanisms can endure beyond any single institution, deployment, or era.

In public technical terms, that long-horizon goal is framed as recursive verification, self-checking, and self-refinement that can be instrumented rather than treated as a black box.

This is an ambition and a research direction, not a claim of achieved sentience. The active public standard remains the same: architecture claims, evidence claims, and shipped code must stay distinguishable.

## Current Mainline

- Canonical code path: [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)
- Architecture line: `INSTNCT`
- Current English recipe candidate on `main`: [`v4.2/english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/english_1024n_18w.py) (`8` ticks, triangle-derived `2 add / 1 flip / 5 decay`)
- Canonical public evidence summary: [Validated Findings](Validated-Findings)
- Mainline implementation includes per-neuron `theta` / `decay` and nonnegative charge dynamics.

Anything not actually shipped in that code path must be labeled as a **Validated finding** or **Experimental branch**, not as a live default.

## Strongest Current Signals

| Topic | Label | Why it matters |
|---|---|---|
| Charge ReLU | Current mainline | Nonnegative charge dynamics already ship in the forward path |
| triangle-derived `2 add / 1 flip / 5 decay` schedule | Validated finding | Promoted into the current English recipe candidate on `main`, not `graph.py` defaults |
| sign+mag + magnitude resample | Validated finding | Best edge-quality result in the current edge-format sweep; not yet promoted into the current recipe candidate |
| voltage medium leak schedule | Validated finding | Strongest schedule result so far |
| mixed 18-worker swarm | Experimental branch | Main active build target, not current mainline |

See [Validated Findings](Validated-Findings) for the full evidence board.

<details>
<summary>Status Taxonomy</summary>

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: an experiment-backed result that has not been promoted into the canonical code path yet.
- **Experimental branch**: an active build target or design direction, not a live default.

If code and docs disagree, **code wins for Current mainline**.

</details>

## 5-Minute Proof

Run the same checks used to keep the public repo aligned with the code:

```bash
python -m compileall v4.2 tools
python v4.2/tests/test_model.py
python tools/check_public_surface.py
```

## Start Here

- [INSTNCT Architecture](SWG-v4.2-Architecture) — architecture line and public status
- [Project Timeline](Release-Notes) — current snapshot, major turns, retirements, and key terms
- [`README.md`](https://github.com/VRAXION/VRAXION/blob/main/README.md) — repo front door
- [Validated Findings](Validated-Findings) — canonical evidence summary
- [Engineering Protocol](Engineering) — run contract and evidence discipline
