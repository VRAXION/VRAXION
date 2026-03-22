<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# VRAXION Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-instnct-spiral.png" alt="INSTNCT spiral logo" width="320">
</p>

**Architecture line:** `INSTNCT`

VRAXION is building **INSTNCT**: a gradient-free self-wiring architecture that learns by changing its own graph instead of using backpropagation through a fixed layer stack.

This page explains the active technical line in plain terms: what the system is, what is actually shipped on `main`, and what makes the current architecture different from a fixed-topology model.

## At a Glance

- **Current mainline:** [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py) is the canonical shipped code path.
- The learnable object is the hidden directed graph plus co-evolved per-neuron `theta` / `decay`.
- If a result is not yet shipped in code, it belongs under **Validated finding** or **Experimental branch**, not under a live default.

## Use This Page When

- you need the current architecture in one screen
- you need to know what is fixed, what is learnable, and what persists across ticks
- you need the shipped facts before reading experimental evidence

## What Makes INSTNCT Different

Most neural systems learn by adjusting lots of weights inside a fixed topology. INSTNCT changes that. Here, the thing being learned is the hidden graph itself.

Input enters through fixed random projections, moves through a self-wiring hidden graph, and is read out through another fixed projection. The graph changes by mutation + selection, while neurons keep charge/state across ticks. In short: the model learns structure and state dynamics, not just layer weights.

## Architecture In One Screen

```text
input -> input_projection -> hidden ternary graph -> output_projection -> output
              persistent charge/state across ticks
```

Mutation-selection loop at a glance:

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/instnct-at-a-glance-training.png" alt="Mutation-selection training loop at a glance" width="740">
</p>

- `input_projection` and `output_projection` are fixed random projections.
- The hidden graph is directed, ternary, and can rewire itself over time.
- Per-neuron `theta` and `decay` are co-evolved with the graph.
- Charge/state persists across ticks instead of resetting after one pass.
- Training happens by mutation + selection, not backpropagation through the graph.

## Current Shipped Facts

| Topic | Current state |
|---|---|
| Canonical code path | [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py) |
| Current English recipe candidate on `main` | [`v4.2/english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/english_1024n_18w.py) (`8` ticks, triangle-derived `2 add / 1 flip / 5 decay`) |
| Shipped constants on `main` | `THRESHOLD = 0.5`, `INJ_SCALE = 3.0`, `DRIVE = 0.6` |
| Mainline runtime behavior | per-neuron `theta` / `decay` and nonnegative charge dynamics |

## What Is Fixed vs Learnable

| Component | What it does |
|---|---|
| `input_projection` | Fixed random projection |
| `output_projection` | Fixed random projection |
| Hidden-to-hidden mask | Learnable graph structure |
| `theta` | Learnable per-neuron firing threshold |
| `decay` | Learnable per-neuron decay rate |
| Charge / state | Runtime state that changes while the model runs |

## Evidence Around This Line

| Topic | Why it matters | Status |
|---|---|---|
| Charge ReLU | Nonnegative charge now ships in the mainline forward path | Current mainline |
| triangle-derived `2 add / 1 flip / 5 decay` schedule | The current English recipe candidate already uses the winning fixed schedule line | Validated finding |
| sign+mag + magnitude resample | Best edge-format quality result so far: `18.69%` at `155` edges (`q=0.121`), but still unpromoted | Validated finding |
| voltage medium leak schedule | Strongest schedule result so far: `22.11%` peak / `21.46%` plateau | Validated finding |
| mixed 18-worker swarm | Current next build target for English training | Experimental branch |

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
